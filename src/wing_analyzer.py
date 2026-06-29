"""
Модуль для анализа крыла по двум STEP файлам (верхняя и нижняя поверхности)
Версия: 2.3 — очистка дублирования, полная функциональность.
"""

import os
import sys
import math
import numpy as np
from datetime import datetime

# PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QLabel, QSpinBox, QFileDialog,
    QTextEdit, QProgressBar, QMessageBox, QGroupBox, QDialog,
    QComboBox, QGridLayout, QRadioButton, QButtonGroup, QCheckBox, QDoubleSpinBox,
    QInputDialog, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal

# Matplotlib
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# 3D поддержка
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    MATPLOTLIB_3D = True
except ImportError:
    MATPLOTLIB_3D = False

# pythonocc-core
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    OCC_SUPPORT = True
except ImportError:
    OCC_SUPPORT = False

# PyVista
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.special import comb


# ============================================================
#  Утилиты CST (общие)
# ============================================================
class CSTUtils:
    """Статические методы для CST-преобразования."""
    @staticmethod
    def bernstein_poly(n, i, x):
        return comb(n, i) * (x**i) * ((1 - x)**(n - i))

    @staticmethod
    def shape(x, coeffs, N1=0.5, N2=1.0, dz=0.0):
        x = np.asarray(x)
        n = len(coeffs) - 1
        shape_arr = np.zeros_like(x)
        x_clipped = np.clip(x, 0, 1)
        for i in range(n + 1):
            shape_arr += coeffs[i] * CSTUtils.bernstein_poly(n, i, x_clipped)
        term1 = x_clipped ** N1
        term2 = (1 - x_clipped) ** N2
        result = term1 * term2 * shape_arr + x_clipped * dz
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def fit(x, y, degree=5, N1=0.5, N2=1.0):
        n = degree
        initial_coeffs = np.zeros(n + 1)
        initial_coeffs[0] = 1.0
        def residuals(coeffs):
            y_pred = CSTUtils.shape(x, coeffs, N1, N2)
            return y_pred - y
        result = least_squares(residuals, initial_coeffs, bounds=(-10, 10))
        return result.x


# Псевдонимы для обратной совместимости в коде визуализаторов
CSTAirfoil = CSTUtils
WingSectionConnector = CSTUtils


# ============================================================
#  Инструменты для обработки сечений
# ============================================================
class SectionTools:
    """Методы для обработки точек профиля."""
    @staticmethod
    def get_mid_tail_point(points, percentile=98.0):
        if len(points) < 3:
            return None, 0.0, 0.0
        le_idx = np.argmin(points[:, 0])
        le_point = points[le_idx]
        shifted = points - le_point
        distances = np.linalg.norm(shifted, axis=1)
        tail_indices = np.argsort(distances)[-2:]
        tail_points = points[tail_indices]
        mid_tail = np.mean(tail_points, axis=0)
        direction = mid_tail - le_point
        angle = np.arctan2(direction[1], direction[0])
        chord_len = np.linalg.norm(direction)
        return mid_tail, angle, chord_len

    @staticmethod
    def connect_wing_section(points):
        """Соединяет точки в замкнутый контур профиля."""
        if len(points) < 3:
            return points, [[] for _ in points], False, "Недостаточно точек (<3)", None, None
        points = np.unique(np.round(points, decimals=6), axis=0)
        n = len(points)
        if n < 3:
            return points, [[] for _ in points], False, "После очистки < 3 точек", None, None
        le_idx = np.argmin(points[:, 0])
        le_point = points[le_idx]
        shifted = points - le_point
        distances = np.linalg.norm(shifted, axis=1)
        te_idx = np.argmax(distances)
        te_point = shifted[te_idx]
        chord_angle = np.arctan2(te_point[1], te_point[0]) if distances[te_idx] > 1e-6 else 0.0
        cos_a, sin_a = np.cos(-chord_angle), np.sin(-chord_angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        aligned = shifted @ rot_matrix.T
        upper_mask = aligned[:, 1] >= -1e-6
        lower_mask = aligned[:, 1] <= 1e-6
        upper_points = points[upper_mask]
        lower_points = points[lower_mask]
        upper_aligned = aligned[upper_mask]
        lower_aligned = aligned[lower_mask]
        if len(upper_points) < 2 or len(lower_points) < 2:
            return SectionTools._fallback_angular_sort(points) + (None, None)
        upper_sorted = upper_points[np.argsort(upper_aligned[:, 0])]
        lower_sorted = lower_points[np.argsort(lower_aligned[:, 0])][::-1]
        if len(upper_sorted) > 0 and upper_sorted[0][0] > 1e-6:
            upper_sorted = np.vstack(([le_point], upper_sorted))
        if len(lower_sorted) > 0 and lower_sorted[0][0] > 1e-6:
            lower_sorted = np.vstack(([le_point], lower_sorted))
        contour_points = np.vstack((upper_sorted, lower_sorted))
        if len(contour_points) < 3:
            return points, [[] for _ in points], False, "Контур содержит менее 3 точек", upper_sorted, lower_sorted
        n = len(contour_points)
        connections = [[] for _ in range(n)]
        for i in range(n - 1):
            connections[i].append(i + 1)
            connections[i + 1].append(i)
        connections[0].append(n - 1)
        connections[n - 1].append(0)
        is_closed = all(len(conn) == 2 for conn in connections)
        return contour_points, connections, is_closed, "Контур построен", upper_sorted, lower_sorted

    @staticmethod
    def _fallback_angular_sort(points):
        center = np.mean(points, axis=0)
        vectors = points - center
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        distances_sort = np.linalg.norm(vectors, axis=1)
        order = np.lexsort((distances_sort, angles))
        connections = [[] for _ in range(len(points))]
        n = len(order)
        for i in range(n):
            prev = order[(i - 1) % n]
            next_ = order[(i + 1) % n]
            curr = order[i]
            connections[curr].append(prev)
            connections[curr].append(next_)
        for i in range(len(connections)):
            connections[i] = list(set(connections[i]))
        return points, connections, False, "Запасной режим", None, None

    @staticmethod
    def auto_approximate_section(upper_points, lower_points, approx_points=200, degree=5):
        """
        Автоматическая аппроксимация сечения методом CST (нос) + PCHIP (хвост).
        Возвращает (upper_control_points, lower_control_points) в нормализованных координатах.
        """
        if upper_points is None or lower_points is None:
            return None, None
        if len(upper_points) < 10 or len(lower_points) < 10:
            return None, None
        try:
            le_point = upper_points[0]
            all_points = np.vstack([upper_points, lower_points])
            mid_tail, angle, chord_len = SectionTools.get_mid_tail_point(all_points)
            if mid_tail is not None:
                chord_length = chord_len
                chord_angle = angle
            else:
                chord_vector = upper_points[-1] - le_point
                chord_length = np.linalg.norm(chord_vector)
                chord_angle = np.arctan2(chord_vector[1], chord_vector[0]) if chord_length > 1e-6 else 0.0
            cos_a, sin_a = np.cos(-chord_angle), np.sin(-chord_angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            shifted_upper = upper_points - le_point
            rotated_upper = shifted_upper @ rot_matrix.T
            x_upper = rotated_upper[:, 0] / chord_length
            y_upper = rotated_upper[:, 1] / chord_length
            shifted_lower = lower_points - le_point
            rotated_lower = shifted_lower @ rot_matrix.T
            x_lower = rotated_lower[:, 0] / chord_length
            y_lower = rotated_lower[:, 1] / chord_length
            x_upper = np.clip(x_upper, 0, 1)
            x_lower = np.clip(x_lower, 0, 1)
            sort_idx_upper = np.argsort(x_upper)
            x_upper = x_upper[sort_idx_upper]
            y_upper = y_upper[sort_idx_upper]
            sort_idx_lower = np.argsort(x_lower)
            x_lower = x_lower[sort_idx_lower]
            y_lower = y_lower[sort_idx_lower]
            x_nose_main = 0.15
            N1, N2 = 0.5, 1.0
            # Добавляем граничную точку x_nose_main, если её нет
            if not np.any(np.isclose(x_upper, x_nose_main, atol=1e-4)):
                idx = np.searchsorted(x_upper, x_nose_main)
                if 0 < idx < len(x_upper):
                    x1, x2 = x_upper[idx-1], x_upper[idx]
                    y1, y2 = y_upper[idx-1], y_upper[idx]
                    y_at_boundary = y1 + (y2 - y1) * (x_nose_main - x1) / (x2 - x1)
                    x_upper = np.insert(x_upper, idx, x_nose_main)
                    y_upper = np.insert(y_upper, idx, y_at_boundary)
            if not np.any(np.isclose(x_lower, x_nose_main, atol=1e-4)):
                idx = np.searchsorted(x_lower, x_nose_main)
                if 0 < idx < len(x_lower):
                    x1, x2 = x_lower[idx-1], x_lower[idx]
                    y1, y2 = y_lower[idx-1], y_lower[idx]
                    y_at_boundary = y1 + (y2 - y1) * (x_nose_main - x1) / (x2 - x1)
                    x_lower = np.insert(x_lower, idx, x_nose_main)
                    y_lower = np.insert(y_lower, idx, y_at_boundary)
            # CST для носовой части (0..0.15)
            upper_coeffs = None
            lower_coeffs = None
            u_mask_cst = (x_upper <= x_nose_main) & (x_upper >= 0)
            x_u_cst = x_upper[u_mask_cst]
            y_u_cst = y_upper[u_mask_cst]
            if len(x_u_cst) >= 3 and np.all(np.isfinite(x_u_cst)) and np.all(np.isfinite(y_u_cst)):
                _, unique_idx = np.unique(x_u_cst, return_index=True)
                x_u_cst_unique = x_u_cst[unique_idx]
                y_u_cst_unique = y_u_cst[unique_idx]
                if len(x_u_cst_unique) >= 3:
                    try:
                        upper_coeffs = CSTUtils.fit(x_u_cst_unique, y_u_cst_unique, degree, N1, N2)
                    except Exception:
                        pass
            l_mask_cst = (x_lower <= x_nose_main) & (x_lower >= 0)
            x_l_cst = x_lower[l_mask_cst]
            y_l_cst = y_lower[l_mask_cst]
            if len(x_l_cst) >= 3 and np.all(np.isfinite(x_l_cst)) and np.all(np.isfinite(y_l_cst)):
                _, unique_idx = np.unique(x_l_cst, return_index=True)
                x_l_cst_unique = x_l_cst[unique_idx]
                y_l_cst_unique = y_l_cst[unique_idx]
                if len(x_l_cst_unique) >= 3:
                    try:
                        lower_coeffs = CSTUtils.fit(x_l_cst_unique, y_l_cst_unique, degree, N1, N2)
                    except Exception:
                        pass
            # PCHIP для хвостовой части (0.15..1)
            upper_interpolator = None
            lower_interpolator = None
            MIN_POINTS_TAIL = 4
            u_mask_sp = (x_upper >= x_nose_main) & (x_upper <= 1)
            if np.sum(u_mask_sp) >= MIN_POINTS_TAIL:
                try:
                    x_u_sp = x_upper[u_mask_sp]
                    y_u_sp = y_upper[u_mask_sp]
                    _, unique_idx = np.unique(x_u_sp, return_index=True)
                    x_u_sp = x_u_sp[unique_idx]
                    y_u_sp = y_u_sp[unique_idx]
                    if not np.any(np.isclose(x_u_sp, x_nose_main, atol=1e-6)) and upper_coeffs is not None:
                        y_nose_u = CSTUtils.shape(np.array([x_nose_main]), upper_coeffs, N1, N2)[0]
                        if np.isfinite(y_nose_u):
                            x_u_sp = np.concatenate(([x_nose_main], x_u_sp))
                            y_u_sp = np.concatenate(([y_nose_u], y_u_sp))
                    if len(x_u_sp) >= 4 and np.all(np.diff(x_u_sp) > 0):
                        upper_interpolator = PchipInterpolator(x_u_sp, y_u_sp)
                except Exception:
                    pass
            l_mask_sp = (x_lower >= x_nose_main) & (x_lower <= 1)
            if np.sum(l_mask_sp) >= MIN_POINTS_TAIL:
                try:
                    x_l_sp = x_lower[l_mask_sp]
                    y_l_sp = y_lower[l_mask_sp]
                    _, unique_idx = np.unique(x_l_sp, return_index=True)
                    x_l_sp = x_l_sp[unique_idx]
                    y_l_sp = y_l_sp[unique_idx]
                    if not np.any(np.isclose(x_l_sp, x_nose_main, atol=1e-6)) and lower_coeffs is not None:
                        y_nose_l = CSTUtils.shape(np.array([x_nose_main]), lower_coeffs, N1, N2)[0]
                        if np.isfinite(y_nose_l):
                            x_l_sp = np.concatenate(([x_nose_main], x_l_sp))
                            y_l_sp = np.concatenate(([y_nose_l], y_l_sp))
                    if len(x_l_sp) >= 4 and np.all(np.diff(x_l_sp) > 0):
                        lower_interpolator = PchipInterpolator(x_l_sp, y_l_sp)
                except Exception:
                    pass
            # Генерация плотной сетки
            points_per_surface = max(3, approx_points // 2)
            x_dense = np.linspace(0, 1, 2000)
            y_upper_dense = np.zeros_like(x_dense)
            y_lower_dense = np.zeros_like(x_dense)
            for i, x_val in enumerate(x_dense):
                if x_val <= x_nose_main and upper_coeffs is not None:
                    try:
                        y_val = CSTUtils.shape(np.array([x_val]), upper_coeffs, N1, N2)[0]
                        y_upper_dense[i] = y_val if np.isfinite(y_val) else np.nan
                    except Exception:
                        y_upper_dense[i] = np.nan
                elif upper_interpolator is not None:
                    try:
                        y_val = upper_interpolator(x_val)
                        y_upper_dense[i] = y_val if np.isfinite(y_val) else np.nan
                    except Exception:
                        y_upper_dense[i] = np.nan
                else:
                    y_upper_dense[i] = np.nan
                if x_val <= x_nose_main and lower_coeffs is not None:
                    try:
                        y_val = CSTUtils.shape(np.array([x_val]), lower_coeffs, N1, N2)[0]
                        y_lower_dense[i] = y_val if np.isfinite(y_val) else np.nan
                    except Exception:
                        y_lower_dense[i] = np.nan
                elif lower_interpolator is not None:
                    try:
                        y_val = lower_interpolator(x_val)
                        y_lower_dense[i] = y_val if np.isfinite(y_val) else np.nan
                    except Exception:
                        y_lower_dense[i] = np.nan
                else:
                    y_lower_dense[i] = np.nan
            valid_mask = ~(np.isnan(y_upper_dense) | np.isnan(y_lower_dense))
            if np.sum(valid_mask) < points_per_surface * 2:
                return None, None
            x_valid = x_dense[valid_mask]
            y_upper_valid = y_upper_dense[valid_mask]
            y_lower_valid = y_lower_dense[valid_mask]
            # Передискретизация по длине дуги
            try:
                x_upper_uniform, y_upper_uniform = SectionTools.arc_length_parameterization(x_valid, y_upper_valid, points_per_surface)
                x_lower_uniform, y_lower_uniform = SectionTools.arc_length_parameterization(x_valid, y_lower_valid, points_per_surface)
            except Exception:
                return None, None
            upper_control = np.column_stack((x_upper_uniform, y_upper_uniform))
            lower_control = np.column_stack((x_lower_uniform, y_lower_uniform))
            return upper_control, lower_control
        except Exception:
            return None, None

    @staticmethod
    def arc_length_parameterization(x, y, num_points):
        if len(x) < 2 or len(y) < 2:
            return x, y
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cumulative_length[-1]
        if total_length < 1e-10:
            return x, y
        target_lengths = np.linspace(0, total_length, num_points)
        x_uniform = np.interp(target_lengths, cumulative_length, x)
        y_uniform = np.interp(target_lengths, cumulative_length, y)
        return x_uniform, y_uniform


# ============================================================
#  Экспорт .056
# ============================================================
class ExportUtils:
    @staticmethod
    def export_to_056(sections_data, filename=None, parent_window=None):
        """Экспорт в формат .056 с использованием аппроксимированных точек."""
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                None, "Сохранить файл .056", "wing_profiles.056", "056 files (*.056)")
            if not filename:
                return False
        try:
            approx_points = 200
            if parent_window and hasattr(parent_window, 'current_approx_points'):
                approx_points = parent_window.current_approx_points
            if approx_points % 2 != 0:
                approx_points += 1
            with open(filename, 'w', encoding='utf-8') as f:
                # Находим корневое сечение
                root_section = None
                XTEF0 = 1.0
                XLEF = YLEF = XTEF = YTEF = 0.0
                for sec in sections_data:
                    if len(sec) >= 6:
                        pts, _, _, upper_raw, lower_raw, _ = sec[:6]
                        if pts is not None and len(pts) >= 8:
                            root_section = pts
                            le_root = upper_raw[0] if len(upper_raw) > 0 else np.array([0., 0.])
                            mid_tail, angle, chord_len = SectionTools.get_mid_tail_point(pts)
                            if mid_tail is not None:
                                XTEF0 = chord_len
                                XLEF, YLEF = le_root
                                XTEF, YTEF = mid_tail
                            else:
                                shifted_root = pts - le_root
                                distances_root = np.linalg.norm(shifted_root, axis=1)
                                te_idx_root = np.argmax(distances_root)
                                te_point_root = shifted_root[te_idx_root]
                                XTEF0 = np.linalg.norm(te_point_root) if distances_root[te_idx_root] > 1e-6 else 1.0
                                XLEF, YLEF = le_root
                                XTEF, YTEF = pts[te_idx_root]
                            break
                if root_section is None:
                    return False
                NSF = sum(1 for s in sections_data if len(s) >= 6 and s[0] is not None and len(s[0]) >= 8)
                f.write("<  NSEC  >\n")
                f.write(f" {NSF:5d}.\n")
                for sec_idx, sec in enumerate(sections_data, 1):
                    if len(sec) >= 8:
                        raw_points, _, _, upper_raw, lower_raw, Z, upper_approx, lower_approx = sec
                    else:
                        raw_points, _, _, upper_raw, lower_raw, Z = sec[:6]
                        upper_approx = None
                        lower_approx = None
                    if upper_raw is None or lower_raw is None:
                        continue
                    if len(upper_raw) < 3 or len(lower_raw) < 3:
                        continue
                    le_local = upper_raw[0] if len(upper_raw) > 0 else np.array([0.0, 0.0])
                    shifted = raw_points - le_local
                    distances = np.linalg.norm(shifted, axis=1)
                    chord_len = np.max(distances) if np.max(distances) > 1e-6 else 1.0
                    te_idx = np.argmax(distances)
                    te_point = shifted[te_idx]
                    chord_angle = np.arctan2(te_point[1], te_point[0]) if distances[te_idx] > 1e-6 else 0.0
                    XLE = le_local[0] - XLEF
                    YLE = le_local[1] - YLEF
                    EPSIL = np.degrees(chord_angle)
                    cos_a, sin_a = np.cos(-chord_angle), np.sin(-chord_angle)
                    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    if upper_approx is not None and lower_approx is not None and len(upper_approx) > 0:
                        # Используем аппроксимированные точки (без перестановки)
                        XU = lower_approx[:, 0]
                        YU = lower_approx[:, 1]
                        XL = upper_approx[:, 0]
                        YL = upper_approx[:, 1]
                        sort_upper = np.argsort(XU)
                        XU = XU[sort_upper]
                        YU = YU[sort_upper]
                        sort_lower = np.argsort(XL)
                        XL = XL[sort_lower]
                        YL = YL[sort_lower]
                        # Расчет TRAIL, SLOPT, XSING на основе сырых точек
                        shifted_upper_raw = upper_raw - le_local
                        rotated_upper_raw = shifted_upper_raw @ rot_matrix.T
                        XU_raw_norm = rotated_upper_raw[:, 0] / chord_len
                        YU_raw_norm = rotated_upper_raw[:, 1] / chord_len
                        shifted_lower_raw = lower_raw - le_local
                        rotated_lower_raw = shifted_lower_raw @ rot_matrix.T
                        XL_raw_norm = rotated_lower_raw[:, 0] / chord_len
                        YL_raw_norm = rotated_lower_raw[:, 1] / chord_len
                        sort_upper_raw = np.argsort(XU_raw_norm)
                        XU_raw_sorted = XU_raw_norm[sort_upper_raw]
                        YU_raw_sorted = YU_raw_norm[sort_upper_raw]
                        sort_lower_raw = np.argsort(XL_raw_norm)
                        XL_raw_sorted = XL_raw_norm[sort_lower_raw]
                        YL_raw_sorted = YL_raw_norm[sort_lower_raw]
                        TRAIL = 8.0
                        try:
                            if len(XU_raw_sorted) >= 5 and len(XL_raw_sorted) >= 5:
                                n_tail = max(3, len(XU_raw_sorted) // 10)
                                coeffs_upper = np.polyfit(XU_raw_sorted[-n_tail:], YU_raw_sorted[-n_tail:], 1)
                                coeffs_lower = np.polyfit(XL_raw_sorted[-n_tail:], YL_raw_sorted[-n_tail:], 1)
                                angle_upper = np.arctan(coeffs_upper[0])
                                angle_lower = np.arctan(coeffs_lower[0])
                                TRAIL = np.clip(np.degrees(abs(angle_upper - angle_lower)), 2.0, 45.0)
                        except Exception:
                            pass
                        SLOPT = 0.0
                        try:
                            if len(XU_raw_sorted) >= 10 and len(XL_raw_sorted) >= 10:
                                x_start = 0.75
                                x_tail = np.linspace(x_start, 1.0, 20)
                                f_upper = interp1d(XU_raw_sorted, YU_raw_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
                                y_upper_tail = f_upper(x_tail)
                                f_lower = interp1d(XL_raw_sorted, YL_raw_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
                                y_lower_tail = f_lower(x_tail)
                                y_camber = (y_upper_tail + y_lower_tail) / 2
                                coeffs_camber = np.polyfit(x_tail, y_camber, 1)
                                SLOPT = coeffs_camber[0]
                        except Exception:
                            pass
                        XSING = 0.005
                        YSING = 0.0
                        try:
                            points_upper_raw = np.column_stack((XU_raw_sorted, YU_raw_sorted))
                            points_lower_raw = np.column_stack((XL_raw_sorted, YL_raw_sorted))
                            all_points_raw = np.vstack([points_upper_raw, points_lower_raw])
                            le_idx = np.argmin(all_points_raw[:, 0])
                            x_le_rel = all_points_raw[le_idx, 0]
                            if x_le_rel < 1e-6:
                                positive_mask = all_points_raw[:, 0] > 1e-6
                                positive_points = all_points_raw[positive_mask]
                                if len(positive_points) > 0:
                                    min_positive_idx = np.argmin(positive_points[:, 0])
                                    x_le_rel = positive_points[min_positive_idx, 0]
                            nose_mask = (all_points_raw[:, 0] >= x_le_rel) & (all_points_raw[:, 0] <= x_le_rel + 0.1)
                            nose_points = all_points_raw[nose_mask]
                            if len(nose_points) >= 3:
                                A = np.c_[2 * nose_points[:, 0], 2 * nose_points[:, 1], np.ones(len(nose_points))]
                                b = nose_points[:, 0]**2 + nose_points[:, 1]**2
                                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                                x0_rel, y0_rel, r2 = sol
                                R_rel = np.sqrt(max(0, r2 + x0_rel**2 + y0_rel**2))
                                if 0.001 < R_rel < 0.2:
                                    xsing_rel = max(0.001, x_le_rel + 0.5 * R_rel)
                                else:
                                    xsing_rel = max(0.001, x_le_rel)
                            else:
                                xsing_rel = max(0.001, x_le_rel)
                            chord_len_m = chord_len / 1000.0
                            XSING = xsing_rel * chord_len_m
                        except Exception:
                            chord_len_m = chord_len / 1000.0
                            XSING = 0.005 * chord_len_m
                        YSYM = 0
                        THICK = 1.00000
                        FSEC = 1.00000
                        NU = len(XU)
                        NL = len(XL)
                        def _fixlen(n: float, w: int) -> str:
                            if np.isnan(n) or np.isinf(n):
                                n = 0.0
                            n = float(n)
                            if abs(n) >= 10**w:
                                n = 9.99999
                            s = f"{n:.{w-2}f}"
                            if len(s) > w:
                                s = s[:w]
                            return s.ljust(w, '0')
                        f.write(f"<   Z    ><   XLE  ><   YLE  ><  CHORD >< THICK  >< EPSIL  ><  FSEC  >  SEC {sec_idx}\n")
                        f.write(f" {_fixlen(Z/1000,8)}  {_fixlen(XLE/1000,8)}  {_fixlen(YLE/1000,8)}  {_fixlen(chord_len/1000,8)}  {_fixlen(THICK,8)}  {_fixlen(EPSIL,8)}  {_fixlen(FSEC,8)}\n")
                        f.write("<  YSYM  ><   NU   ><   NL   ><\n")
                        f.write(f" {_fixlen(YSYM,8)}  {_fixlen(float(NU),8)}  {_fixlen(float(NL),8)} \n")
                        f.write("<  XSING >< YSING  >< TRAIL  >< SLOPT  >\n")
                        f.write(f" {_fixlen(XSING,8)}  {_fixlen(YSING,8)}  {_fixlen(TRAIL,8)}  {_fixlen(SLOPT,8)}\n")
                        f.write("<   XU   ><   YU   >\n")
                        for x, y in zip(XU, YU):
                            f.write(f" {_fixlen(x,8)}  {_fixlen(y,8)}\n")
                        f.write("<   XL   ><   YL   >\n")
                        for x, y in zip(XL, YL):
                            f.write(f" {_fixlen(x,8)}  {_fixlen(y,8)}\n")
                        continue
                    # Стандартный метод (без аппроксимации)
                    shifted_upper = upper_raw - le_local
                    rotated_upper = shifted_upper @ rot_matrix.T
                    XU_norm_raw = rotated_upper[:, 0] / chord_len
                    YU_norm_raw = rotated_upper[:, 1] / chord_len
                    shifted_lower = lower_raw - le_local
                    rotated_lower = shifted_lower @ rot_matrix.T
                    XL_norm_raw = rotated_lower[:, 0] / chord_len
                    YL_norm_raw = rotated_lower[:, 1] / chord_len
                    sort_upper = np.argsort(XU_norm_raw)
                    XU_norm_raw = XU_norm_raw[sort_upper]
                    YU_norm_raw = YU_norm_raw[sort_upper]
                    sort_lower = np.argsort(XL_norm_raw)
                    XL_norm_raw = XL_norm_raw[sort_lower]
                    YL_norm_raw = YL_norm_raw[sort_lower]
                    if np.mean(YU_norm_raw) < 0:
                        YU_norm_raw = -YU_norm_raw
                    if np.mean(YL_norm_raw) > 0:
                        YL_norm_raw = -YL_norm_raw
                    TRAIL = 8.0
                    try:
                        if len(XU_norm_raw) >= 5 and len(XL_norm_raw) >= 5:
                            n_tail = max(3, len(XU_norm_raw) // 10)
                            coeffs_upper = np.polyfit(XU_norm_raw[-n_tail:], YU_norm_raw[-n_tail:], 1)
                            coeffs_lower = np.polyfit(XL_norm_raw[-n_tail:], YL_norm_raw[-n_tail:], 1)
                            angle_upper = np.arctan(coeffs_upper[0])
                            angle_lower = np.arctan(coeffs_lower[0])
                            TRAIL = np.degrees(abs(angle_upper - angle_lower))
                            TRAIL = np.clip(TRAIL, 2.0, 45.0)
                    except Exception:
                        pass
                    SLOPT = 0.0
                    try:
                        if len(XU_norm_raw) >= 10 and len(XL_norm_raw) >= 10:
                            x_start = 0.75
                            x_tail = np.linspace(x_start, 1.0, 20)
                            f_upper = interp1d(XU_norm_raw, YU_norm_raw, kind='linear', bounds_error=False, fill_value='extrapolate')
                            y_upper_tail = f_upper(x_tail)
                            f_lower = interp1d(XL_norm_raw, YL_norm_raw, kind='linear', bounds_error=False, fill_value='extrapolate')
                            y_lower_tail = f_lower(x_tail)
                            y_camber = (y_upper_tail + y_lower_tail) / 2
                            coeffs_camber = np.polyfit(x_tail, y_camber, 1)
                            SLOPT = coeffs_camber[0]
                    except Exception:
                        pass
                    XSING = 0.005
                    YSING = 0.0
                    try:
                        points_upper_raw = np.column_stack((XU_norm_raw, YU_norm_raw))
                        points_lower_raw = np.column_stack((XL_norm_raw, YL_norm_raw))
                        all_points_raw = np.vstack([points_upper_raw, points_lower_raw])
                        le_idx = np.argmin(all_points_raw[:, 0])
                        x_le_rel = all_points_raw[le_idx, 0]
                        if x_le_rel < 1e-6:
                            positive_mask = all_points_raw[:, 0] > 1e-6
                            positive_points = all_points_raw[positive_mask]
                            if len(positive_points) > 0:
                                min_positive_idx = np.argmin(positive_points[:, 0])
                                x_le_rel = positive_points[min_positive_idx, 0]
                        nose_mask = (all_points_raw[:, 0] >= x_le_rel) & (all_points_raw[:, 0] <= x_le_rel + 0.1)
                        nose_points = all_points_raw[nose_mask]
                        if len(nose_points) >= 3:
                            A = np.c_[2 * nose_points[:, 0], 2 * nose_points[:, 1], np.ones(len(nose_points))]
                            b = nose_points[:, 0]**2 + nose_points[:, 1]**2
                            sol = np.linalg.lstsq(A, b, rcond=None)[0]
                            x0_rel, y0_rel, r2 = sol
                            R_rel = np.sqrt(max(0, r2 + x0_rel**2 + y0_rel**2))
                            if 0.001 < R_rel < 0.2:
                                xsing_rel = max(0.001, x_le_rel + 0.5 * R_rel)
                            else:
                                xsing_rel = max(0.001, x_le_rel)
                        else:
                            xsing_rel = max(0.001, x_le_rel)
                        chord_len_m = chord_len / 1000.0
                        XSING = xsing_rel * chord_len_m
                    except Exception:
                        chord_len_m = chord_len / 1000.0
                        XSING = 0.005 * chord_len_m
                    YSYM = 0
                    THICK = 1.00000
                    FSEC = 1.00000
                    points_per_surface = approx_points // 2
                    x_dense = np.linspace(0, 1, 2000)
                    YU_dense = np.interp(x_dense, XU_norm_raw, YU_norm_raw)
                    YL_dense = np.interp(x_dense, XL_norm_raw, YL_norm_raw)
                    YU_dense[0] = 0.0
                    YL_dense[0] = 0.0
                    YU_dense[-1] = YU_dense[-2] * 0.5 if len(YU_dense) > 1 else 0.0
                    YL_dense[-1] = YL_dense[-2] * 0.5 if len(YL_dense) > 1 else 0.0
                    XU_uniform, YU_uniform = SectionTools.arc_length_parameterization(x_dense, YU_dense, points_per_surface)
                    XL_uniform, YL_uniform = SectionTools.arc_length_parameterization(x_dense, YL_dense, points_per_surface)
                    XU = XU_uniform
                    YU = YU_uniform
                    XL = XL_uniform
                    YL = YL_uniform
                    NU = len(XU)
                    NL = len(XL)
                    def _fixlen(n: float, w: int) -> str:
                        if np.isnan(n) or np.isinf(n):
                            n = 0.0
                        n = float(n)
                        if abs(n) >= 10**w:
                            n = 9.99999
                        s = f"{n:.{w-2}f}"
                        if len(s) > w:
                            s = s[:w]
                        return s.ljust(w, '0')
                    f.write(f"<   Z    ><   XLE  ><   YLE  ><  CHORD >< THICK  >< EPSIL  ><  FSEC  >  SEC {sec_idx}\n")
                    f.write(f" {_fixlen(Z/1000,8)}  {_fixlen(XLE/1000,8)}  {_fixlen(YLE/1000,8)}  {_fixlen(chord_len/1000,8)}  {_fixlen(THICK,8)}  {_fixlen(EPSIL,8)}  {_fixlen(FSEC,8)}\n")
                    f.write("<  YSYM  ><   NU   ><   NL   ><\n")
                    f.write(f" {_fixlen(YSYM,8)}  {_fixlen(float(NU),8)}  {_fixlen(float(NL),8)} \n")
                    f.write("<  XSING >< YSING  >< TRAIL  >< SLOPT  >\n")
                    f.write(f" {_fixlen(XSING,8)}  {_fixlen(YSING,8)}  {_fixlen(TRAIL,8)}  {_fixlen(SLOPT,8)}\n")
                    f.write("<   XU   ><   YU   >\n")
                    for x, y in zip(XU, YU):
                        f.write(f" {_fixlen(x,8)}  {_fixlen(y,8)}\n")
                    f.write("<   XL   ><   YL   >\n")
                    for x, y in zip(XL, YL):
                        f.write(f" {_fixlen(x,8)}  {_fixlen(y,8)}\n")
                return True
        except Exception as e:
            QMessageBox.critical(None, "Ошибка", f"Ошибка экспорта: {str(e)}")
            return False


# ============================================================
#  Визуализаторы (полностью исходный код, с заменами на CSTUtils/SectionTools)
# ============================================================
class HybridApproximationViewer(QDialog):
    def __init__(self, points, connections, is_closed,
                 upper_points=None, lower_points=None,
                 approx_points=200, parent=None):
        super().__init__(parent)
        self.points = points
        self.connections = connections
        self.is_closed = is_closed
        self.upper_points_raw = upper_points
        self.lower_points_raw = lower_points
        self.approx_points = approx_points
        self.normalized = False
        self.upper_coeffs = None
        self.lower_coeffs = None
        self.upper_interpolator = None
        self.lower_interpolator = None
        self.chord_len = None
        self.le_point = None
        self.chord_angle = None
        self.x_nose_main = 0.6
        self.x_nose_tip = 0.15
        self.degree = 5
        self.nose_density_multiplier = 1
        self.mid_tail_point = None
        self.rotation_source = "хвостовой сгусток"
        self.upper_control_points = None
        self.lower_control_points = None
        self.all_approx_points = None
        self.init_ui()
        self.compute_approximation()

    @staticmethod
    def arc_length_parameterization(x, y, num_points):
        return SectionTools.arc_length_parameterization(x, y, num_points)

    def init_ui(self):
        self.setWindowTitle(f"Гибрид: CST (нос) + PCHIP (хвост) | Опорных точек: {self.approx_points}")
        self.resize(1200, 900)
        layout = QVBoxLayout()
        info_text = f"Запрошено опорных точек: {self.approx_points} | "
        info_text += f"Исходных точек - верх: {len(self.upper_points_raw) if self.upper_points_raw is not None else 0}, низ: {len(self.lower_points_raw) if self.lower_points_raw is not None else 0}"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e0f7fa;")
        layout.addWidget(info_label)
        coord_layout = QHBoxLayout()
        self.abs_btn = QPushButton("Абсолютные координаты (мм)")
        self.abs_btn.setEnabled(False)
        self.rel_btn = QPushButton("Относительные координаты (0..1)")
        self.abs_btn.clicked.connect(self.switch_to_absolute)
        self.rel_btn.clicked.connect(self.switch_to_relative)
        coord_layout.addWidget(self.abs_btn)
        coord_layout.addWidget(self.rel_btn)
        self.control_points_btn = QPushButton("Скрыть опорные точки")
        self.control_points_btn.setCheckable(True)
        self.control_points_btn.setChecked(True)
        self.control_points_btn.clicked.connect(self.toggle_control_points)
        coord_layout.addWidget(self.control_points_btn)
        coord_layout.addStretch()
        layout.addLayout(coord_layout)
        self.fig = Figure(figsize=(16, 12), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        control_group = QGroupBox("Настройки аппроксимации")
        control_layout = QGridLayout()
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(3, 9)
        self.degree_spin.setValue(5)
        self.degree_spin.valueChanged.connect(self.update_approximation)
        control_layout.addWidget(QLabel("Степень CST:"), 0, 0)
        control_layout.addWidget(self.degree_spin, 0, 1)
        self.approx_points_spin = QSpinBox()
        self.approx_points_spin.setRange(10, 1000)
        self.approx_points_spin.setValue(self.approx_points)
        self.approx_points_spin.setSingleStep(10)
        self.approx_points_spin.valueChanged.connect(self.update_approx_points)
        control_layout.addWidget(QLabel("Опорных точек:"), 0, 2)
        control_layout.addWidget(self.approx_points_spin, 0, 3)
        info_label_fixed = QLabel("Параметры носовой части: плотность=1, граница=0.15 (фиксированы)")
        info_label_fixed.setStyleSheet("color: #666; font-style: italic;")
        control_layout.addWidget(info_label_fixed, 1, 0, 1, 4)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        button_layout = QHBoxLayout()
        export_btn = QPushButton("Экспорт аппроксимации")
        export_btn.clicked.connect(self.export_approximation)
        button_layout.addWidget(export_btn)
        save_img_btn = QPushButton("Сохранить изображение")
        save_img_btn.clicked.connect(self.save_image)
        button_layout.addWidget(save_img_btn)
        reset_btn = QPushButton("Сброс вида")
        reset_btn.clicked.connect(self.reset_view)
        button_layout.addWidget(reset_btn)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        info_group = QGroupBox("Информация об аппроксимации")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        self.setLayout(layout)

    def update_approx_points(self):
        self.approx_points = self.approx_points_spin.value()
        self.generate_control_points()
        self.plot_approximation()

    def toggle_control_points(self):
        if self.control_points_btn.isChecked():
            self.control_points_btn.setText("Скрыть опорные точки")
        else:
            self.control_points_btn.setText("Показать опорные точки")
        self.plot_approximation()

    def compute_approximation(self):
        if self.upper_points_raw is None or self.lower_points_raw is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных ordered_upper/lower")
            return
        if len(self.upper_points_raw) < 10 or len(self.lower_points_raw) < 10:
            QMessageBox.warning(self, "Предупреждение",
                                f"Недостаточно исходных точек (верх: {len(self.upper_points_raw)}, низ: {len(self.lower_points_raw)})")
            return
        self.le_point = self.upper_points_raw[0]
        all_points = np.vstack([self.upper_points_raw, self.lower_points_raw])
        mid_tail, angle, chord_len = SectionTools.get_mid_tail_point(all_points)
        if mid_tail is not None:
            self.chord_len = chord_len
            self.chord_angle = angle
            self.mid_tail_point = mid_tail
            self.rotation_source = "середина хвостового сгустка"
        else:
            chord_vector = self.upper_points_raw[-1] - self.le_point
            self.chord_len = np.linalg.norm(chord_vector)
            self.chord_angle = np.arctan2(chord_vector[1], chord_vector[0]) if self.chord_len > 1e-6 else 0.0
            self.rotation_source = "хорда"
        self.update_approximation()

    def update_approximation(self):
        if self.upper_points_raw is None or self.lower_points_raw is None:
            return
        self.degree = self.degree_spin.value()
        N1 = 0.5
        N2 = 1.0
        try:
            shifted_upper = self.upper_points_raw - self.le_point
            shifted_lower = self.lower_points_raw - self.le_point
            cos_a, sin_a = np.cos(-self.chord_angle), np.sin(-self.chord_angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_upper = shifted_upper @ rot_matrix.T
            rotated_lower = shifted_lower @ rot_matrix.T
            x_upper = rotated_upper[:, 0] / self.chord_len
            y_upper = rotated_upper[:, 1] / self.chord_len
            x_lower = rotated_lower[:, 0] / self.chord_len
            y_lower = rotated_lower[:, 1] / self.chord_len
            sort_idx_upper = np.argsort(x_upper)
            x_upper = x_upper[sort_idx_upper]
            y_upper = y_upper[sort_idx_upper]
            sort_idx_lower = np.argsort(x_lower)
            x_lower = x_lower[sort_idx_lower]
            y_lower = y_lower[sort_idx_lower]
            if not np.any(np.isclose(x_upper, self.x_nose_main, atol=1e-4)):
                idx = np.searchsorted(x_upper, self.x_nose_main)
                if 0 < idx < len(x_upper):
                    x1, x2 = x_upper[idx-1], x_upper[idx]
                    y1, y2 = y_upper[idx-1], y_upper[idx]
                    y_at_boundary = y1 + (y2 - y1) * (self.x_nose_main - x1) / (x2 - x1)
                    x_upper = np.insert(x_upper, idx, self.x_nose_main)
                    y_upper = np.insert(y_upper, idx, y_at_boundary)
            if not np.any(np.isclose(x_lower, self.x_nose_main, atol=1e-4)):
                idx = np.searchsorted(x_lower, self.x_nose_main)
                if 0 < idx < len(x_lower):
                    x1, x2 = x_lower[idx-1], x_lower[idx]
                    y1, y2 = y_lower[idx-1], y_lower[idx]
                    y_at_boundary = y1 + (y2 - y1) * (self.x_nose_main - x1) / (x2 - x1)
                    x_lower = np.insert(x_lower, idx, self.x_nose_main)
                    y_lower = np.insert(y_lower, idx, y_at_boundary)
            u_mask_cst = x_upper <= self.x_nose_main
            x_u_cst = x_upper[u_mask_cst]
            y_u_cst = y_upper[u_mask_cst]
            l_mask_cst = x_lower <= self.x_nose_main
            x_l_cst = x_lower[l_mask_cst]
            y_l_cst = y_lower[l_mask_cst]
            if len(x_u_cst) >= 3:
                self.upper_coeffs = CSTUtils.fit(x_u_cst, y_u_cst, self.degree, N1, N2)
            else:
                self.upper_coeffs = None
            if len(x_l_cst) >= 3:
                self.lower_coeffs = CSTUtils.fit(x_l_cst, y_l_cst, self.degree, N1, N2)
            else:
                self.lower_coeffs = None
            MIN_POINTS_TAIL = 4
            u_mask_sp = x_upper >= self.x_nose_main
            if np.sum(u_mask_sp) >= MIN_POINTS_TAIL:
                x_u_sp = x_upper[u_mask_sp]
                y_u_sp = y_upper[u_mask_sp]
                unique_idx = np.unique(x_u_sp, return_index=True)[1]
                x_u_sp = x_u_sp[unique_idx]
                y_u_sp = y_u_sp[unique_idx]
                if not np.any(np.isclose(x_u_sp, self.x_nose_main, atol=1e-6)) and self.upper_coeffs is not None:
                    y_nose_u = CSTUtils.shape(np.array([self.x_nose_main]), self.upper_coeffs, N1, N2)[0]
                    x_u_sp = np.concatenate(([self.x_nose_main], x_u_sp))
                    y_u_sp = np.concatenate(([y_nose_u], y_u_sp))
                if np.all(np.diff(x_u_sp) > 0):
                    self.upper_interpolator = PchipInterpolator(x_u_sp, y_u_sp)
                else:
                    sort_idx = np.argsort(x_u_sp)
                    self.upper_interpolator = PchipInterpolator(x_u_sp[sort_idx], y_u_sp[sort_idx])
            else:
                self.upper_interpolator = None
            l_mask_sp = x_lower >= self.x_nose_main
            if np.sum(l_mask_sp) >= MIN_POINTS_TAIL:
                x_l_sp = x_lower[l_mask_sp]
                y_l_sp = y_lower[l_mask_sp]
                unique_idx = np.unique(x_l_sp, return_index=True)[1]
                x_l_sp = x_l_sp[unique_idx]
                y_l_sp = y_l_sp[unique_idx]
                if not np.any(np.isclose(x_l_sp, self.x_nose_main, atol=1e-6)) and self.lower_coeffs is not None:
                    y_nose_l = CSTUtils.shape(np.array([self.x_nose_main]), self.lower_coeffs, N1, N2)[0]
                    x_l_sp = np.concatenate(([self.x_nose_main], x_l_sp))
                    y_l_sp = np.concatenate(([y_nose_l], y_l_sp))
                if np.all(np.diff(x_l_sp) > 0):
                    self.lower_interpolator = PchipInterpolator(x_l_sp, y_l_sp)
                else:
                    sort_idx = np.argsort(x_l_sp)
                    self.lower_interpolator = PchipInterpolator(x_l_sp[sort_idx], y_l_sp[sort_idx])
            else:
                self.lower_interpolator = None
            self.generate_approximation_points()
            info = self.generate_info_text()
            self.info_text.setText(info)
            self.plot_approximation()
        except Exception as e:
            self.info_text.setText(f"Ошибка: {str(e)}")

    def generate_control_points(self):
        if not hasattr(self, 'all_approx_points') or self.all_approx_points is None:
            return
        x_all = self.all_approx_points['x']
        y_upper_all = self.all_approx_points['y_upper']
        y_lower_all = self.all_approx_points['y_lower']
        points_per_surface = max(3, self.approx_points // 2)
        if len(x_all) >= 3 and len(y_upper_all) >= 3:
            x_upper_uniform, y_upper_uniform = self.arc_length_parameterization(x_all, y_upper_all, points_per_surface)
            self.upper_control_points = np.column_stack((x_upper_uniform, y_upper_uniform))
        else:
            self.upper_control_points = None
        if len(x_all) >= 3 and len(y_lower_all) >= 3:
            x_lower_uniform, y_lower_uniform = self.arc_length_parameterization(x_all, y_lower_all, points_per_surface)
            self.lower_control_points = np.column_stack((x_lower_uniform, y_lower_uniform))
        else:
            self.lower_control_points = None

    def generate_approximation_points(self):
        if (self.upper_coeffs is None and self.upper_interpolator is None) or \
           (self.lower_coeffs is None and self.lower_interpolator is None):
            x_total = np.linspace(0, 1, max(200, self.approx_points * 2))
            self.all_approx_points = {
                'x': x_total,
                'y_upper': np.zeros_like(x_total),
                'y_lower': np.zeros_like(x_total)
            }
            self.generate_control_points()
            return
        try:
            total_points = max(2000, self.approx_points * 10)
            t = np.linspace(0, 1, total_points)
            p = 0.5
            x_total = t ** p
            tip_mask = x_total <= self.x_nose_tip
            if np.sum(tip_mask) < total_points * 0.2:
                tip_points = np.linspace(0, self.x_nose_tip, total_points // 5)
                x_total = np.sort(np.unique(np.concatenate([x_total, tip_points])))
            nose_main_mask = x_total <= self.x_nose_main
            tail_mask = x_total >= self.x_nose_main
            x_nose_main_arr = x_total[nose_main_mask]
            x_tail = x_total[tail_mask]
            y_upper = np.full_like(x_total, np.nan)
            y_lower = np.full_like(x_total, np.nan)
            if self.upper_coeffs is not None and len(x_nose_main_arr) > 0:
                y_upper_nose = CSTUtils.shape(x_nose_main_arr, self.upper_coeffs, N1=0.5, N2=1.0)
                y_upper[nose_main_mask] = y_upper_nose
                y_boundary_upper = CSTUtils.shape(np.array([self.x_nose_main]), self.upper_coeffs, N1=0.5, N2=1.0)[0]
            else:
                y_boundary_upper = None
            if self.lower_coeffs is not None and len(x_nose_main_arr) > 0:
                y_lower_nose = CSTUtils.shape(x_nose_main_arr, self.lower_coeffs, N1=0.5, N2=1.0)
                y_lower[nose_main_mask] = y_lower_nose
                y_boundary_lower = CSTUtils.shape(np.array([self.x_nose_main]), self.lower_coeffs, N1=0.5, N2=1.0)[0]
            else:
                y_boundary_lower = None
            if self.upper_interpolator is not None and len(x_tail) > 0:
                try:
                    y_tail_upper = self.upper_interpolator(x_tail)
                    y_upper[tail_mask] = y_tail_upper
                    if y_boundary_upper is not None and len(x_tail) > 0:
                        y_upper[x_total >= self.x_nose_main][0] = y_boundary_upper
                except Exception:
                    if y_boundary_upper is not None and len(x_tail) > 0:
                        y_upper[tail_mask] = y_boundary_upper
            if self.lower_interpolator is not None and len(x_tail) > 0:
                try:
                    y_tail_lower = self.lower_interpolator(x_tail)
                    y_lower[tail_mask] = y_tail_lower
                    if y_boundary_lower is not None and len(x_tail) > 0:
                        y_lower[x_total >= self.x_nose_main][0] = y_boundary_lower
                except Exception:
                    if y_boundary_lower is not None and len(x_tail) > 0:
                        y_lower[tail_mask] = y_boundary_lower
            valid_mask = ~(np.isnan(y_upper) | np.isnan(y_lower))
            x_total = x_total[valid_mask]
            y_upper = y_upper[valid_mask]
            y_lower = y_lower[valid_mask]
            self.all_approx_points = {
                'x': x_total,
                'y_upper': y_upper,
                'y_lower': y_lower
            }
            self.generate_control_points()
        except Exception as e:
            x_total = np.linspace(0, 1, max(200, self.approx_points * 2))
            self.all_approx_points = {
                'x': x_total,
                'y_upper': np.zeros_like(x_total),
                'y_lower': np.zeros_like(x_total)
            }
            self.generate_control_points()

    def generate_info_text(self):
        info = f" АППРОКСИМАЦИЯ\n{'='*40}\n"
        info += f"Режим: {'ОТНОСИТЕЛЬНЫЕ координаты (хорда=1)' if self.normalized else 'АБСОЛЮТНЫЕ координаты (мм)'}\n"
        info += f"Поворот относительно: {self.rotation_source}\n"
        info += f"Угол поворота: {np.degrees(self.chord_angle):.2f}°\n"
        info += f"Длина вектора: {self.chord_len:.3f} мм\n"
        info += f"Запрошено опорных точек: {self.approx_points}\n"
        info += f"МЕТОД РАСПРЕДЕЛЕНИЯ: РАВНОМЕРНО ПО ДЛИНЕ ДУГИ\n"
        if self.upper_control_points is not None and self.lower_control_points is not None:
            total_control = len(self.upper_control_points) + len(self.lower_control_points)
            info += f"Фактически опорных точек: {total_control}\n"
            info += f" • Верх: {len(self.upper_control_points)} точек (равномерно по длине дуги)\n"
            info += f" • Низ: {len(self.lower_control_points)} точек (равномерно по длине дуги)\n"
        if hasattr(self, 'all_approx_points') and self.all_approx_points is not None:
            info += f"Точек для плавной кривой: {len(self.all_approx_points['x'])}\n"
            nose_mask = self.all_approx_points['x'] <= self.x_nose_main
            tip_mask = self.all_approx_points['x'] <= self.x_nose_tip
            tail_mask = self.all_approx_points['x'] >= self.x_nose_main
            nose_points = np.sum(nose_mask)
            tip_points = np.sum(tip_mask)
            tail_points = np.sum(tail_mask)
            info += f" • CST (нос 0–{self.x_nose_main:.2f}): {nose_points} точек\n"
            info += f" • Особо тщательная аппроксимация носка (0–{self.x_nose_tip:.2f}): {tip_points} точек\n"
        info += f"CST степень: {self.degree}\n"
        info += f"Плотность носка: 1x (фиксировано)\n"
        info += f"Граница носка: 0.15 (фиксировано)\n"
        info += f"Сглаживание носка: Выключено\n"
        info += f"PCHIP: кусочно-кубическая интерполяция\n\n"
        if self.upper_coeffs is not None:
            coeffs_str = ', '.join(f'{c:.3f}' for c in self.upper_coeffs[:3])
            if len(self.upper_coeffs) > 3:
                coeffs_str += f"... (всего {len(self.upper_coeffs)} коэф.)"
            info += f"Коэффициенты CST верх: {coeffs_str}\n"
        if self.lower_coeffs is not None:
            coeffs_str = ', '.join(f'{c:.3f}' for c in self.lower_coeffs[:3])
            if len(self.lower_coeffs) > 3:
                coeffs_str += f"... (всего {len(self.lower_coeffs)} коэф.)"
            info += f"Коэффициенты CST низ: {coeffs_str}\n"
        if hasattr(self, 'all_approx_points') and self.all_approx_points is not None and len(self.all_approx_points['x']) > 0:
            thickness = self.all_approx_points['y_upper'] - self.all_approx_points['y_lower']
            max_thickness_idx = np.argmax(thickness)
            max_thickness = thickness[max_thickness_idx]
            max_thickness_x = self.all_approx_points['x'][max_thickness_idx]
            info += f"Макс. толщина: {max_thickness:.4f} при x={max_thickness_x:.3f}\n"
        return info

    def plot_approximation(self):
        if not hasattr(self, 'all_approx_points') or self.all_approx_points is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Нет данных аппроксимации", transform=self.ax.transAxes, ha='center', va='center')
            self.canvas.draw()
            return
        self.ax.clear()
        x_all = self.all_approx_points['x']
        y_upper_all = self.all_approx_points['y_upper']
        y_lower_all = self.all_approx_points['y_lower']
        if self.normalized:
            x_approx = x_all
            y_approx_upper = y_upper_all
            y_approx_lower = y_lower_all
            upper_control = self.upper_control_points
            lower_control = self.lower_control_points
            x_label = "X (0..1)"
            y_label = "Y (относительно хорды)"
            title_suffix = f" (Относительные координаты, поворот: {self.rotation_source})"
        else:
            cos_a, sin_a = np.cos(self.chord_angle), np.sin(self.chord_angle)
            x_rot = x_all * self.chord_len
            y_rot_upper = y_upper_all * self.chord_len
            y_rot_lower = y_lower_all * self.chord_len
            x_approx = x_rot * cos_a - y_rot_upper * sin_a + self.le_point[0]
            y_approx_upper = x_rot * sin_a + y_rot_upper * cos_a + self.le_point[1]
            x_approx_lower = x_rot * cos_a - y_rot_lower * sin_a + self.le_point[0]
            y_approx_lower = x_rot * sin_a + y_rot_lower * cos_a + self.le_point[1]
            if self.upper_control_points is not None:
                upper_control = []
                for point in self.upper_control_points:
                    x_norm, y_norm = point
                    x_rot_pt = x_norm * self.chord_len
                    y_rot_pt = y_norm * self.chord_len
                    x_abs = x_rot_pt * cos_a - y_rot_pt * sin_a + self.le_point[0]
                    y_abs = x_rot_pt * sin_a + y_rot_pt * cos_a + self.le_point[1]
                    upper_control.append([x_abs, y_abs])
                upper_control = np.array(upper_control)
            else:
                upper_control = None
            if self.lower_control_points is not None:
                lower_control = []
                for point in self.lower_control_points:
                    x_norm, y_norm = point
                    x_rot_pt = x_norm * self.chord_len
                    y_rot_pt = y_norm * self.chord_len
                    x_abs = x_rot_pt * cos_a - y_rot_pt * sin_a + self.le_point[0]
                    y_abs = x_rot_pt * sin_a + y_rot_pt * cos_a + self.le_point[1]
                    lower_control.append([x_abs, y_abs])
                lower_control = np.array(lower_control)
            else:
                lower_control = None
            x_label = "X (мм)"
            y_label = "Y (мм)"
            title_suffix = f" (Абсолютные координаты, поворот: {self.rotation_source})"
        self.ax.plot(x_approx, y_approx_upper, 'r-', linewidth=2.5, alpha=0.9, label='нижняя поверхность')
        self.ax.plot(x_approx if self.normalized else x_approx_lower,
                     y_approx_lower if self.normalized else y_approx_lower,
                     'b-', linewidth=2.5, alpha=0.9, label='верхняя поверхность')
        if self.control_points_btn.isChecked():
            if upper_control is not None and len(upper_control) > 0:
                self.ax.scatter(upper_control[:, 0], upper_control[:, 1],
                                c='darkred', s=30, alpha=0.7, marker='o',
                                edgecolors='black', linewidth=0.5, label='Опорные точки (низ)')
            if lower_control is not None and len(lower_control) > 0:
                self.ax.scatter(lower_control[:, 0], lower_control[:, 1],
                                c='darkblue', s=30, alpha=0.7, marker='o',
                                edgecolors='black', linewidth=0.5, label='Опорные точки (верх)')
        self.ax.legend(loc='best')
        total_approx_points = len(x_all)
        info_text = f"Точек для кривой: {total_approx_points}"
        upper_count = len(upper_control) if upper_control is not None else 0
        lower_count = len(lower_control) if lower_control is not None else 0
        info_text += f"\nОпорных точек: верх {upper_count}, низ {lower_count} (запрошено: {self.approx_points})"
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        self.ax.set_title(f' аппроксимация профиля{title_suffix}\n{info_text}')
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        if self.normalized:
            all_x = x_approx
            all_y = np.concatenate([y_approx_upper, y_approx_lower])
        else:
            all_x = np.concatenate([x_approx, x_approx_lower])
            all_y = np.concatenate([y_approx_upper, y_approx_lower])
        if len(all_x) > 0:
            margin = 0.9
            x_min, x_max = all_x.min(), all_x.max()
            y_min, y_max = all_y.min(), all_y.max()
            x_range = x_max - x_min or 1
            y_range = y_max - y_min or 1
            self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        self.canvas.draw()

    def export_approximation(self):
        if not hasattr(self, 'all_approx_points'):
            QMessageBox.warning(self, "Предупреждение", "Сначала вычислите аппроксимацию")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить аппроксимацию", "approximation.csv", "CSV (*.csv)")
        if file_path:
            try:
                data = []
                for i in range(len(self.all_approx_points['x'])):
                    if self.normalized:
                        x, y = self.all_approx_points['x'][i], self.all_approx_points['y_upper'][i]
                    else:
                        cos_a, sin_a = np.cos(self.chord_angle), np.sin(self.chord_angle)
                        x_rot = self.all_approx_points['x'][i] * self.chord_len
                        y_rot = self.all_approx_points['y_upper'][i] * self.chord_len
                        x = x_rot * cos_a - y_rot * sin_a + self.le_point[0]
                        y = x_rot * sin_a + y_rot * cos_a + self.le_point[1]
                    data.append([x, y, "upper", "approx"])
                for i in range(len(self.all_approx_points['x'])):
                    if self.normalized:
                        x, y = self.all_approx_points['x'][i], self.all_approx_points['y_lower'][i]
                    else:
                        cos_a, sin_a = np.cos(self.chord_angle), np.sin(self.chord_angle)
                        x_rot = self.all_approx_points['x'][i] * self.chord_len
                        y_rot = self.all_approx_points['y_lower'][i] * self.chord_len
                        x = x_rot * cos_a - y_rot * sin_a + self.le_point[0]
                        y = x_rot * sin_a + y_rot * cos_a + self.le_point[1]
                    data.append([x, y, "lower", "approx"])
                if self.upper_control_points is not None:
                    for point in self.upper_control_points:
                        if self.normalized:
                            data.append([point[0], point[1], "upper", "control"])
                        else:
                            x_norm, y_norm = point
                            x_rot = x_norm * self.chord_len
                            y_rot = y_norm * self.chord_len
                            cos_a, sin_a = np.cos(self.chord_angle), np.sin(self.chord_angle)
                            x_abs = x_rot * cos_a - y_rot * sin_a + self.le_point[0]
                            y_abs = x_rot * sin_a + y_rot * cos_a + self.le_point[1]
                            data.append([x_abs, y_abs, "upper", "control"])
                if self.lower_control_points is not None:
                    for point in self.lower_control_points:
                        if self.normalized:
                            data.append([point[0], point[1], "lower", "control"])
                        else:
                            x_norm, y_norm = point
                            x_rot = x_norm * self.chord_len
                            y_rot = y_norm * self.chord_len
                            cos_a, sin_a = np.cos(self.chord_angle), np.sin(self.chord_angle)
                            x_abs = x_rot * cos_a - y_rot * sin_a + self.le_point[0]
                            y_abs = x_rot * sin_a + y_rot * cos_a + self.le_point[1]
                            data.append([x_abs, y_abs, "lower", "control"])
                np.savetxt(file_path, data, delimiter=',',
                           header='X,Y,Surface,Type',
                           fmt='%.6f,%.6f,%s,%s',
                           comments='')
                QMessageBox.information(self, "Экспорт", f"Сохранено {len(data)} точек в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))

    def switch_to_absolute(self):
        self.normalized = False
        self.abs_btn.setEnabled(False)
        self.rel_btn.setEnabled(True)
        self.plot_approximation()

    def switch_to_relative(self):
        self.normalized = True
        self.abs_btn.setEnabled(True)
        self.rel_btn.setEnabled(False)
        self.plot_approximation()

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "approximation_plot.png", "PNG (*.png)")
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Успех", f"Сохранено: {file_path}")

    def reset_view(self):
        self.ax.autoscale()
        self.canvas.draw()


class MatplotlibSectionViewer(QDialog):
    def __init__(self, section_data, section_number, total_sections, cut_axis_info=None,
                 upper_points=None, lower_points=None, parent=None, raw_points=None):
        super().__init__(parent)
        self.section_data = section_data
        self.section_number = section_number
        self.total_sections = total_sections
        self.cut_axis_info = cut_axis_info or "Неизвестно"
        self.upper_points = upper_points
        self.lower_points = lower_points
        self.raw_points = raw_points
        self.normalized = False
        self.parent_window = parent
        self.mid_tail_point = None
        self.rotation_angle = 0.0
        self.chord_length = 1.0
        self.init_ui()
        self.plot_section()

    def init_ui(self):
        points, connections, is_closed = self.section_data
        point_count = len(points) if points is not None else 0
        title = f"Сечение {self.section_number + 1} из {self.total_sections}"
        if is_closed:
            title += " (ЗАМКНУТЫЙ)"
        else:
            title += " (НЕЗАМКНУТЫЙ)"
        self.setWindowTitle(title)
        self.resize(1000, 800)
        layout = QVBoxLayout()
        raw_count = len(self.raw_points) if self.raw_points is not None else 0
        info_text = f"Точек: {point_count} (соединенных) | Сырых точек: {raw_count} | "
        info_text += "Замкнутый" if is_closed else "Незамкнутый"
        if self.cut_axis_info:
            info_text += f" | Ось: {self.cut_axis_info}"
        if self.upper_points is not None:
            info_text += f" | Верх: {len(self.upper_points)}"
        if self.lower_points is not None:
            info_text += f" | Низ: {len(self.lower_points)}"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(info_label)
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        display_group = QGroupBox("Отображение")
        display_layout = QHBoxLayout()
        self.show_raw_btn = QPushButton("Скрыть сырые точки")
        self.show_raw_btn.setCheckable(True)
        self.show_raw_btn.setChecked(True)
        self.show_raw_btn.clicked.connect(self.toggle_raw_points)
        display_layout.addWidget(self.show_raw_btn)
        self.show_connections_btn = QPushButton("Скрыть связи")
        self.show_connections_btn.setCheckable(True)
        self.show_connections_btn.setChecked(True)
        self.show_connections_btn.clicked.connect(self.toggle_connections)
        display_layout.addWidget(self.show_connections_btn)
        display_layout.addStretch()
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        mode_layout = QHBoxLayout()
        self.abs_btn = QPushButton("Абсолютные координаты")
        self.abs_btn.setEnabled(False)
        self.rel_btn = QPushButton("Относительные (0..1)")
        self.abs_btn.clicked.connect(self.switch_to_absolute)
        self.rel_btn.clicked.connect(self.switch_to_relative)
        mode_layout.addWidget(self.abs_btn)
        mode_layout.addWidget(self.rel_btn)
        layout.addLayout(mode_layout)
        control_layout = QHBoxLayout()
        self.prev_btn = QPushButton("← Предыдущее")
        self.next_btn = QPushButton("Следующее →")
        save_btn = QPushButton("Сохранить изображение")
        reset_btn = QPushButton("Сбросить вид")
        export_btn = QPushButton("Экспорт .056")
        hybrid_btn = QPushButton("CST аппроксимация")
        close_btn = QPushButton("Закрыть")
        self.prev_btn.clicked.connect(self.prev_section)
        self.next_btn.clicked.connect(self.next_section)
        save_btn.clicked.connect(self.save_image)
        reset_btn.clicked.connect(self.reset_view)
        export_btn.clicked.connect(self.export_056)
        hybrid_btn.clicked.connect(self.show_hybrid_approximation)
        close_btn.clicked.connect(self.close)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(save_btn)
        control_layout.addWidget(reset_btn)
        control_layout.addWidget(export_btn)
        control_layout.addWidget(hybrid_btn)
        control_layout.addWidget(close_btn)
        layout.addLayout(control_layout)
        self.setLayout(layout)
        self.update_nav_buttons()

    def toggle_raw_points(self):
        if self.show_raw_btn.isChecked():
            self.show_raw_btn.setText("Скрыть сырые точки")
        else:
            self.show_raw_btn.setText("Показать сырые точки")
        self.plot_section()

    def toggle_connections(self):
        if self.show_connections_btn.isChecked():
            self.show_connections_btn.setText("Скрыть связи")
        else:
            self.show_connections_btn.setText("Показать связи")
        self.plot_section()

    def update_nav_buttons(self):
        self.prev_btn.setEnabled(self.section_number > 0)
        self.next_btn.setEnabled(self.section_number < self.total_sections - 1)

    def prev_section(self):
        self.close()
        if self.parent_window and hasattr(self.parent_window, 'show_section'):
            self.parent_window.show_section(self.section_number - 1)

    def next_section(self):
        self.close()
        if self.parent_window and hasattr(self.parent_window, 'show_section'):
            self.parent_window.show_section(self.section_number + 1)

    def switch_to_absolute(self):
        self.normalized = False
        self.abs_btn.setEnabled(False)
        self.rel_btn.setEnabled(True)
        self.plot_section()

    def switch_to_relative(self):
        self.normalized = True
        self.abs_btn.setEnabled(True)
        self.rel_btn.setEnabled(False)
        self.plot_section()

    def show_hybrid_approximation(self):
        points, connections, is_closed = self.section_data
        if self.upper_points is None or self.lower_points is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных ordered_upper/lower")
            return
        if len(self.upper_points) < 10 or len(self.lower_points) < 10:
            QMessageBox.warning(self, "Предупреждение", "Недостаточно точек для аппроксимации")
            return
        approx_points = self.parent_window.current_approx_points if hasattr(self.parent_window, 'current_approx_points') else 200
        viewer = HybridApproximationViewer(
            points=points,
            connections=connections,
            is_closed=is_closed,
            upper_points=self.upper_points,
            lower_points=self.lower_points,
            approx_points=approx_points,
            parent=self
        )
        viewer.exec()

    def plot_section(self):
        self.ax.clear()
        points, connections, is_closed = self.section_data
        if points is None or len(points) == 0:
            self.ax.text(0.5, 0.5, "Нет данных", transform=self.ax.transAxes, ha='center', va='center')
            self.canvas.draw()
            return
        if self.normalized:
            le_idx = np.argmin(points[:, 0])
            le_point = points[le_idx]
            shifted = points - le_point
            mid_tail, angle, chord_len = SectionTools.get_mid_tail_point(points)
            if mid_tail is not None:
                chord_angle = angle
                self.chord_length = chord_len
                self.rotation_angle = chord_angle
                self.mid_tail_point = mid_tail
            else:
                distances = np.linalg.norm(shifted, axis=1)
                te_idx = np.argmax(distances)
                te_point = shifted[te_idx]
                chord_angle = np.arctan2(te_point[1], te_point[0]) if distances[te_idx] > 1e-6 else 0.0
                self.chord_length = distances[te_idx] if distances[te_idx] > 1e-6 else 1.0
                self.rotation_angle = chord_angle
            cos_a, sin_a = np.cos(-chord_angle), np.sin(-chord_angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = shifted @ rot_matrix.T
            plot_points = rotated / self.chord_length
            plot_raw_points = None
            if self.raw_points is not None and len(self.raw_points) > 0:
                shifted_raw = self.raw_points - le_point
                rotated_raw = shifted_raw @ rot_matrix.T
                plot_raw_points = rotated_raw / self.chord_length
            x_label = "X (0..1)"
            y_label = "Y (относительно хорды)"
            rotation_info = " (поворот: середина хвоста)" if mid_tail is not None else " (поворот: хорда)"
        else:
            plot_points = points
            plot_raw_points = self.raw_points
            x_label = "X (мм)"
            y_label = "Y (мм)"
            rotation_info = ""
        if self.show_raw_btn.isChecked() and plot_raw_points is not None and len(plot_raw_points) > 0:
            self.ax.scatter(plot_raw_points[:, 0], plot_raw_points[:, 1],
                            c='lightgray', s=15, alpha=0.6, marker='o',
                            edgecolors='gray', linewidth=0.3,
                            label=f'Сырые точки ({len(plot_raw_points)})', zorder=1)
        if self.show_connections_btn.isChecked():
            for i in range(len(plot_points)):
                for j in connections[i]:
                    if i < j:
                        x1, y1 = plot_points[i]
                        x2, y2 = plot_points[j]
                        self.ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.5, linewidth=1.5, zorder=2)
        point_colors = []
        point_sizes = []
        for i in range(len(plot_points)):
            count = len(connections[i])
            if count == 2:
                color, size = 'green', 50
            elif count == 1:
                color, size = 'orange', 60
            elif count == 0:
                color, size = 'red', 70
            else:
                color, size = 'purple', 65
            point_colors.append(color)
            point_sizes.append(size)
        self.ax.scatter(plot_points[:, 0], plot_points[:, 1],
                        c=point_colors, s=point_sizes, alpha=0.9,
                        edgecolors='black', linewidth=0.8,
                        label=f'Соединенные точки ({len(plot_points)})', zorder=3)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        title = f'Сечение {self.section_number + 1} ({"ЗАМКНУТЫЙ" if is_closed else "НЕЗАМКНУТЫЙ"}){rotation_info}'
        if self.cut_axis_info:
            title += f'\nОсь: {self.cut_axis_info}'
        if self.normalized:
            title += f'\n(Относительные координаты, угол={np.degrees(self.rotation_angle):.1f}°)'
        self.ax.set_title(title, fontsize=11)
        self.ax.set_xlabel(x_label, fontsize=10)
        self.ax.set_ylabel(y_label, fontsize=10)
        self.ax.legend(loc='best', fontsize=8, framealpha=0.9)
        all_x = []
        all_y = []
        if self.show_raw_btn.isChecked() and plot_raw_points is not None:
            all_x.extend(plot_raw_points[:, 0])
            all_y.extend(plot_raw_points[:, 1])
        all_x.extend(plot_points[:, 0])
        all_y.extend(plot_points[:, 1])
        if len(all_x) > 0:
            margin = 0.9
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_range = x_max - x_min or 1
            y_range = y_max - y_min or 1
            self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        self.canvas.draw_idle()

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить", f"section_{self.section_number + 1}.png", "PNG (*.png)")
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Успех", f"Сохранено: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))

    def reset_view(self):
        self.ax.autoscale()
        self.canvas.draw()

    def export_056(self):
        if hasattr(self, 'parent_window') and self.parent_window and hasattr(self.parent_window, 'sections'):
            success = ExportUtils.export_to_056(self.parent_window.sections, parent_window=self.parent_window)
            if success:
                QMessageBox.information(self, "Экспорт", "Файл .056 успешно сохранён")
            else:
                QMessageBox.warning(self, "Экспорт", "Экспорт отменён")
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных сечений для экспорта")


class Wing3DViewer(QDialog):
    """3D визуализатор крыла с сечениями (matplotlib версия)"""
    def __init__(self, upper_shape, lower_shape, sections_data, parent=None):
        super().__init__(parent)
        self.upper_shape = upper_shape
        self.lower_shape = lower_shape
        self.sections_data = sections_data
        self.current_section_index = 0
        self.section_patches = []
        self.contour_lines = []
        self.init_ui()
        self.plot_3d_view()

    def init_ui(self):
        self.setWindowTitle("3D модель крыла с сечениями (matplotlib)")
        self.resize(1200, 800)
        layout = QVBoxLayout()
        info_frame = QWidget()
        info_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        info_layout = QHBoxLayout(info_frame)
        info_text = f"Количество сечений: {len(self.sections_data)}"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        info_layout.addWidget(info_label)
        info_layout.addStretch()
        self.pyvista_btn = QPushButton("Открыть в PyVista (улучшенная 3D)")
        self.pyvista_btn.clicked.connect(self.open_pyvista_viewer)
        if not PYVISTA_AVAILABLE:
            self.pyvista_btn.setEnabled(False)
            self.pyvista_btn.setToolTip("PyVista не установлен. Установите: pip install pyvista")
        info_layout.addWidget(self.pyvista_btn)
        view_btn_group = QHBoxLayout()
        self.isometric_btn = QPushButton("Изометрия")
        self.isometric_btn.clicked.connect(self.set_isometric_view)
        self.front_btn = QPushButton("Вид спереди")
        self.front_btn.clicked.connect(self.set_front_view)
        self.side_btn = QPushButton("Вид сбоку")
        self.side_btn.clicked.connect(self.set_side_view)
        self.top_btn = QPushButton("Вид сверху")
        self.top_btn.clicked.connect(self.set_top_view)
        view_btn_group.addWidget(self.isometric_btn)
        view_btn_group.addWidget(self.front_btn)
        view_btn_group.addWidget(self.side_btn)
        view_btn_group.addWidget(self.top_btn)
        info_layout.addLayout(view_btn_group)
        layout.addWidget(info_frame)
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)
        control_layout.addWidget(QLabel("Показать сечения:"))
        self.show_all_btn = QPushButton("Все сечения")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.setChecked(True)
        self.show_all_btn.clicked.connect(self.toggle_all_sections)
        control_layout.addWidget(self.show_all_btn)
        self.show_contours_btn = QPushButton("Только контуры")
        self.show_contours_btn.setCheckable(True)
        self.show_contours_btn.setChecked(False)
        self.show_contours_btn.clicked.connect(self.toggle_contours_only)
        control_layout.addWidget(self.show_contours_btn)
        control_layout.addWidget(QLabel("|"))
        control_layout.addWidget(QLabel("Прозрачность:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        control_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("50%")
        control_layout.addWidget(self.alpha_label)
        control_layout.addStretch()
        export_btn = QPushButton("Сохранить изображение")
        export_btn.clicked.connect(self.save_image)
        control_layout.addWidget(export_btn)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        control_layout.addWidget(close_btn)
        layout.addWidget(control_frame)
        self.setLayout(layout)

    def open_pyvista_viewer(self):
        if not PYVISTA_AVAILABLE:
            QMessageBox.warning(self, "Ошибка", "PyVista не установлен.")
            return
        valid_sections = [s for s in self.sections_data if s[0] is not None and len(s[0]) >= 3]
        if len(valid_sections) < 2:
            QMessageBox.warning(self, "Ошибка", "Недостаточно сечений")
            return
        visualizer = Wing3DVisualizerPyVista(
            sections_data=valid_sections,
            upper_shape=self.upper_shape,
            lower_shape=self.lower_shape
        )
        visualizer.visualize()

    def plot_3d_view(self):
        self.ax.clear()
        self.section_patches = []
        self.contour_lines = []
        for i, section in enumerate(self.sections_data):
            if len(section) < 6:
                continue
            points, connections, is_closed, upper_points, lower_points, z_pos = section[:6]
            if points is None or len(points) < 3:
                continue
            points_3d = np.column_stack((points, np.full(len(points), z_pos)))
            for j in range(len(connections)):
                for k in connections[j]:
                    if j < k:
                        x = [points_3d[j, 0], points_3d[k, 0]]
                        y = [points_3d[j, 1], points_3d[k, 1]]
                        z = [points_3d[j, 2], points_3d[k, 2]]
                        line = self.ax.plot(x, y, z, 'b-', linewidth=1.5, alpha=0.7)[0]
                        self.contour_lines.append(line)
            if self.show_all_btn.isChecked() and not self.show_contours_btn.isChecked():
                if is_closed and len(points) >= 3:
                    ordered_points = self._order_points_for_polygon(points, connections)
                    if len(ordered_points) >= 3:
                        points_3d_poly = np.column_stack((ordered_points, np.full(len(ordered_points), z_pos)))
                        poly = Poly3DCollection([points_3d_poly], alpha=self.alpha_slider.value() / 100.0)
                        poly.set_facecolor('cyan')
                        poly.set_edgecolor('blue')
                        poly.set_linewidth(0.5)
                        self.ax.add_collection3d(poly)
                        self.section_patches.append(poly)
        self._draw_guide_lines()
        self.ax.set_xlabel('X (мм)', fontsize=10)
        self.ax.set_ylabel('Y (мм)', fontsize=10)
        self.ax.set_zlabel('Z (мм)', fontsize=10)
        self.ax.set_title('3D модель крыла с сечениями', fontsize=12, fontweight='bold')
        self._auto_scale()
        self.set_isometric_view()
        self.canvas.draw()

    def _order_points_for_polygon(self, points, connections):
        if len(points) == 0:
            return []
        start_idx = np.argmin(points[:, 0])
        ordered = [start_idx]
        visited = set([start_idx])
        current = start_idx
        max_iter = len(points) * 2
        iter_count = 0
        while len(ordered) < len(points) and iter_count < max_iter:
            iter_count += 1
            next_found = False
            for neighbor in connections[current]:
                if neighbor not in visited:
                    ordered.append(neighbor)
                    visited.add(neighbor)
                    current = neighbor
                    next_found = True
                    break
            if not next_found:
                for i in range(len(points)):
                    if i not in visited:
                        ordered.append(i)
                        visited.add(i)
                        current = i
                        break
                else:
                    break
        return np.array([points[i] for i in ordered])

    def _draw_guide_lines(self):
        le_points = []
        te_points = []
        for section in self.sections_data:
            if len(section) < 6:
                continue
            points, connections, is_closed, upper_points, lower_points, z_pos = section[:6]
            if points is None or len(points) < 3:
                continue
            le_idx = np.argmin(points[:, 0])
            le_points.append([points[le_idx, 0], points[le_idx, 1], z_pos])
            te_idx = np.argmax(np.linalg.norm(points - points[le_idx], axis=1))
            te_points.append([points[te_idx, 0], points[te_idx, 1], z_pos])
        if len(le_points) > 1:
            le_points = np.array(le_points)
            self.ax.plot(le_points[:, 0], le_points[:, 1], le_points[:, 2],
                         'r-', linewidth=2, alpha=0.6, label='Передняя кромка')
        if len(te_points) > 1:
            te_points = np.array(te_points)
            self.ax.plot(te_points[:, 0], te_points[:, 1], te_points[:, 2],
                         'g-', linewidth=2, alpha=0.6, label='Задняя кромка')

    def _auto_scale(self):
        all_points = []
        for section in self.sections_data:
            if len(section) < 6:
                continue
            points = section[0]
            if points is not None and len(points) > 0:
                all_points.append(points)
        if all_points:
            all_points = np.vstack(all_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            z_min = min([s[5] for s in self.sections_data if len(s) > 5])
            z_max = max([s[5] for s in self.sections_data if len(s) > 5])
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            mid_z = (z_max + z_min) / 2
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def set_isometric_view(self):
        self.ax.view_init(elev=30, azim=45)
        self.canvas.draw()

    def set_front_view(self):
        self.ax.view_init(elev=0, azim=-90)
        self.canvas.draw()

    def set_side_view(self):
        self.ax.view_init(elev=0, azim=0)
        self.canvas.draw()

    def set_top_view(self):
        self.ax.view_init(elev=90, azim=0)
        self.canvas.draw()

    def toggle_all_sections(self):
        if self.show_all_btn.isChecked():
            self.show_contours_btn.setChecked(False)
        self.plot_3d_view()

    def toggle_contours_only(self):
        if self.show_contours_btn.isChecked():
            self.show_all_btn.setChecked(False)
        else:
            if not self.show_all_btn.isChecked():
                self.show_all_btn.setChecked(True)
        self.plot_3d_view()

    def update_alpha(self, value):
        self.alpha_label.setText(f"{value}%")
        for patch in self.section_patches:
            patch.set_alpha(value / 100.0)
        self.canvas.draw()

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "wing_3d.png", "PNG (*.png)")
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Успех", f"Сохранено: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))


class Wing3DVisualizerPyVista:
    """3D визуализатор крыла с сечениями на основе PyVista."""
    def __init__(self, sections_data=None, upper_shape=None, lower_shape=None):
        self.sections_data = sections_data if sections_data is not None else []
        self.upper_shape = upper_shape
        self.lower_shape = lower_shape
        self.plotter = None
        self.section_color = '#FF6B6B'
        self.le_color = '#4ECDC4'
        self.te_color = '#FFE66D'
        self.mesh_color = 'lightgray'
        self.mesh_opacity = 0.5
        self.section_line_width = 3
        self.edge_line_width = 4

    def _add_legend(self, section_actors, edge_actors):
        legend_entries = []
        if section_actors:
            legend_entries.append(["Сечения", self.section_color])
        if edge_actors:
            legend_entries.append(["Передняя кромка (LE)", self.le_color])
            legend_entries.append(["Задняя кромка (TE)", self.te_color])
        if legend_entries:
            try:
                self.plotter.add_legend(
                    legend_entries,
                    face='rectangle',
                    size=[0.2, 0.2 * len(legend_entries) / 2],
                    loc='upper right'
                )
            except Exception:
                pass

    def _add_text_safe(self, text, position='upper_left', font_size=10, color='black'):
        try:
            self.plotter.add_text(
                text,
                position=position,
                font_size=font_size,
                color=color,
                background_color='white',
                background_opacity=0.7
            )
        except TypeError:
            try:
                self.plotter.add_text(
                    text,
                    position=position,
                    font_size=font_size,
                    color=color
                )
            except Exception:
                pass

    def visualize(self):
        if not PYVISTA_AVAILABLE:
            print("PyVista не установлен.")
            return
        if not self.sections_data:
            print("Нет данных сечений для визуализации")
            return
        valid_sections = sum(1 for data in self.sections_data
                             if len(data) >= 6 and data[0] is not None and len(data[0]) > 2)
        le_points, te_points = self._extract_le_te_points()
        has_le = len(le_points) > 0
        has_te = len(te_points) > 0
        try:
            self._setup_plotter()
            title_parts = []
            if valid_sections > 0:
                title_parts.append(f"{valid_sections} сечений")
            if has_le:
                title_parts.append("передняя кромка")
            if has_te:
                title_parts.append("задняя кромка")
            title = "3D модель крыла" + (f" с {', '.join(title_parts)}" if title_parts else "")
            self.plotter.title = title
            info_lines = [
                f"Сечений: {valid_sections} из {len(self.sections_data)}",
                f"Точек передней кромки: {len(le_points)}",
                f"Точек задней кромки: {len(te_points)}"
            ]
            self._add_text_safe("\n".join(info_lines), position='upper_left', font_size=10, color='black')
            self.plotter.show()
        except Exception:
            raise

    def _setup_plotter(self):
        self.plotter = pv.Plotter(window_size=[1400, 900])
        if self.sections_data:
            all_points = []
            for data in self.sections_data:
                if len(data) < 6:
                    continue
                points = data[0]
                if points is not None and len(points) > 0:
                    all_points.append(points)
            if all_points:
                all_points = np.vstack(all_points)
                x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
                y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
                z_values = [d[5] for d in self.sections_data if len(d) > 5]
                if z_values:
                    z_min, z_max = min(z_values), max(z_values)
                    grid = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
                    self.plotter.add_mesh(grid, color=self.mesh_color, opacity=0.1, style='wireframe')
        section_actors = self._create_section_actors()
        edge_actors = self._create_edge_actors()
        self.plotter.show_axes()
        self.plotter.add_axes(line_width=3, labels_off=False)
        self.plotter.add_bounding_box(color='black', line_width=1, opacity=0.3)
        self._add_legend(section_actors, edge_actors)
        self.plotter.camera_position = 'xy'
        self.plotter.camera.zoom(1.2)

    def _create_section_actors(self):
        section_actors = []
        if not self.sections_data:
            return section_actors
        valid_sections = []
        for i, data in enumerate(self.sections_data):
            if len(data) < 6:
                continue
            points, connections, is_closed, upper_points, lower_points, z_pos = data[:6]
            if self._has_valid_points(points) and len(points) > 2:
                valid_sections.append({
                    'points': points,
                    'z_pos': z_pos,
                    'is_closed': is_closed,
                    'index': i
                })
        if not valid_sections:
            return section_actors
        first_z = valid_sections[0]['z_pos']
        last_z = valid_sections[-1]['z_pos']
        for section_info in valid_sections:
            points = section_info['points']
            z_pos = section_info['z_pos']
            is_closed = section_info['is_closed']
            idx = section_info['index']
            points_3d = np.column_stack((points[:, 0], points[:, 1], np.full(len(points), z_pos)))
            if len(points_3d) > 1:
                poly = pv.PolyData()
                poly.points = points_3d
                lines = []
                for j in range(len(points_3d) - 1):
                    lines.extend([2, j, j + 1])
                if is_closed and len(points_3d) > 2:
                    lines.extend([2, len(points_3d) - 1, 0])
                if lines:
                    poly.lines = lines
                    actor = self.plotter.add_mesh(
                        poly,
                        color=self.section_color,
                        line_width=self.section_line_width,
                        render_lines_as_tubes=True,
                        name=f"wing_section_{idx}"
                    )
                    section_actors.append(actor)
                    if abs(z_pos - first_z) < 1e-6 or abs(z_pos - last_z) < 1e-6:
                        if len(points_3d) > 0:
                            center = np.mean(points_3d, axis=0)
                            try:
                                self.plotter.add_point_labels(
                                    [center],
                                    [f"Z = {z_pos:.1f} мм"],
                                    point_color=self.section_color,
                                    point_size=8,
                                    font_size=10,
                                    shadow=True,
                                    always_visible=True
                                )
                            except Exception:
                                pass
        return section_actors

    def _create_edge_actors(self):
        edge_actors = []
        le_points, te_points = self._extract_le_te_points()
        if len(le_points) > 1:
            le_points_3d = np.array(le_points)
            le_poly = pv.PolyData(le_points_3d)
            lines = [len(le_points_3d)] + list(range(len(le_points_3d)))
            le_poly.lines = lines
            actor = self.plotter.add_mesh(
                le_poly,
                color=self.le_color,
                line_width=self.edge_line_width,
                render_lines_as_tubes=True,
                name="leading_edge"
            )
            edge_actors.append(actor)
            if len(le_points_3d) > 0:
                try:
                    self.plotter.add_point_labels(
                        le_points_3d[[0, -1]],
                        ["LE start", "LE end"],
                        point_color=self.le_color,
                        point_size=5,
                        font_size=8,
                        shadow=True
                    )
                except Exception:
                    pass
        if len(te_points) > 1:
            te_points_3d = np.array(te_points)
            te_poly = pv.PolyData(te_points_3d)
            lines = [len(te_points_3d)] + list(range(len(te_points_3d)))
            te_poly.lines = lines
            actor = self.plotter.add_mesh(
                te_poly,
                color=self.te_color,
                line_width=self.edge_line_width,
                render_lines_as_tubes=True,
                name="trailing_edge"
            )
            edge_actors.append(actor)
            if len(te_points_3d) > 0:
                try:
                    self.plotter.add_point_labels(
                        te_points_3d[[0, -1]],
                        ["TE start", "TE end"],
                        point_color=self.te_color,
                        point_size=5,
                        font_size=8,
                        shadow=True
                    )
                except Exception:
                    pass
        return edge_actors

    def _has_valid_points(self, points) -> bool:
        if points is None:
            return False
        if isinstance(points, (np.ndarray, list)):
            if len(points) == 0:
                return False
            if isinstance(points, np.ndarray):
                return not np.all(np.isnan(points))
            return True
        return False

    def _extract_le_te_points(self):
        le_points = []
        te_points = []
        for data in self.sections_data:
            if len(data) < 6:
                continue
            points, connections, is_closed, upper_points, lower_points, z_pos = data[:6]
            if points is None or len(points) < 3:
                continue
            le_idx = np.argmin(points[:, 0])
            le_point = points[le_idx]
            le_points.append([le_point[0], le_point[1], z_pos])
            le_point_for_dist = np.array([le_point[0], le_point[1]])
            distances = np.linalg.norm(points[:, :2] - le_point_for_dist, axis=1)
            te_idx = np.argmax(distances)
            te_point = points[te_idx]
            te_points.append([te_point[0], te_point[1], z_pos])
        return le_points, te_points


# Точка входа для тестирования
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from wing_analyzer_two_files import TwoFilesMainWindow
    window = TwoFilesMainWindow()
    window.show()
    sys.exit(app.exec())