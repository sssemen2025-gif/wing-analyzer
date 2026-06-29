"""
Модуль для анализа крыла по двум STEP файлам (верхняя и нижняя поверхности)
Версия: 2.2 — без поворота, исправлены ошибки.
"""

import os
import sys
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
    QMessageBox, QProgressBar, QTextEdit, QSpinBox, QComboBox,
    QGridLayout, QMainWindow, QApplication, QWidget, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QThread

from wing_analyzer import (
    OCC_SUPPORT, SectionTools, ExportUtils,
    MatplotlibSectionViewer, HybridApproximationViewer,
    Wing3DViewer, Wing3DVisualizerPyVista, PYVISTA_AVAILABLE
)

if OCC_SUPPORT:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib


class ManualSectionsDialog(QDialog):
    """Диалог для ручного ввода Z координат сечений"""
    def __init__(self, parent=None, min_z=-1000, max_z=1000, current_sections=None):
        super().__init__(parent)
        self.min_z = min_z
        self.max_z = max_z
        self.current_sections = current_sections or []
        self.setWindowTitle("Ручной ввод Z координат сечений")
        self.resize(500, 400)
        layout = QVBoxLayout()
        info = QLabel(f"Диапазон Z: {self.min_z:.1f} до {self.max_z:.1f} мм")
        info.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info)
        input_group = QGroupBox("Координаты Z (мм)")
        input_layout = QVBoxLayout()
        self.coords_input = QTextEdit()
        self.coords_input.setPlaceholderText(
            "Введите Z координаты через запятую или пробел\n"
            "Пример: -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500"
        )
        if self.current_sections:
            coords_str = ", ".join([f"{z:.1f}" for z in self.current_sections])
            self.coords_input.setText(coords_str)
        input_layout.addWidget(self.coords_input)
        btn_layout = QHBoxLayout()
        auto_btn = QPushButton("Автоматически (равномерно)")
        auto_btn.clicked.connect(self.auto_generate)
        btn_layout.addWidget(auto_btn)
        sort_btn = QPushButton("Сортировать")
        sort_btn.clicked.connect(self.sort_coords)
        btn_layout.addWidget(sort_btn)
        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.coords_input.clear)
        btn_layout.addWidget(clear_btn)
        input_layout.addLayout(btn_layout)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        dialog_buttons = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        dialog_buttons.addStretch()
        dialog_buttons.addWidget(ok_btn)
        dialog_buttons.addWidget(cancel_btn)
        layout.addLayout(dialog_buttons)
        self.setLayout(layout)

    def auto_generate(self):
        num, ok = QInputDialog.getInt(
            self, "Количество сечений",
            "Введите количество сечений:",
            len(self.current_sections) or 10, 2, 50
        )
        if ok and num > 1:
            sections = np.linspace(self.min_z, self.max_z, num)
            self.coords_input.setText(", ".join([f"{z:.1f}" for z in sections]))

    def sort_coords(self):
        coords = self.get_coordinates()
        if coords:
            coords.sort()
            self.coords_input.setText(", ".join([f"{z:.1f}" for z in coords]))

    def get_coordinates(self):
        import re
        text = self.coords_input.toPlainText()
        if not text.strip():
            return []
        numbers = re.findall(r'-?\d+\.?\d*', text)
        coords = []
        for num in numbers:
            try:
                z = float(num)
                if self.min_z <= z <= self.max_z:
                    coords.append(z)
                else:
                    QMessageBox.warning(
                        self, "Предупреждение",
                        f"Значение {z} выходит за пределы диапазона [{self.min_z:.0f}, {self.max_z:.0f}]"
                    )
            except ValueError:
                continue
        return sorted(list(set(coords)))


class TwoFilesSliceProcessor:
    """Класс для обработки двух файлов STEP с автоматической аппроксимацией"""
    def __init__(self):
        self.upper_shape = None
        self.lower_shape = None
        self.sections_data = []
        self.bounding_box_upper = None
        self.bounding_box_lower = None
        self.bbox_dimensions_upper = [0, 0, 0]
        self.bbox_dimensions_lower = [0, 0, 0]
        self.common_bbox = {'min': [0, 0, 0], 'max': [0, 0, 0]}

    def load_upper_file(self, file_path):
        if not OCC_SUPPORT:
            return False
        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(file_path)
            if status != 1:
                return False
            reader.TransferRoots()
            self.upper_shape = reader.OneShape()
            if self.upper_shape.IsNull():
                return False
            bbox = self._calculate_bounding_box(self.upper_shape)
            self.bounding_box_upper = bbox
            self.bbox_dimensions_upper = [
                bbox['max'][0] - bbox['min'][0],
                bbox['max'][1] - bbox['min'][1],
                bbox['max'][2] - bbox['min'][2]
            ]
            self._update_common_bbox()
            return True
        except Exception:
            return False

    def load_lower_file(self, file_path):
        if not OCC_SUPPORT:
            return False
        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(file_path)
            if status != 1:
                return False
            reader.TransferRoots()
            self.lower_shape = reader.OneShape()
            if self.lower_shape.IsNull():
                return False
            bbox = self._calculate_bounding_box(self.lower_shape)
            self.bounding_box_lower = bbox
            self.bbox_dimensions_lower = [
                bbox['max'][0] - bbox['min'][0],
                bbox['max'][1] - bbox['min'][1],
                bbox['max'][2] - bbox['min'][2]
            ]
            self._update_common_bbox()
            return True
        except Exception:
            return False

    def _calculate_bounding_box(self, shape):
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return {'min': (xmin, ymin, zmin), 'max': (xmax, ymax, zmax)}

    def _update_common_bbox(self):
        if self.bounding_box_upper and self.bounding_box_lower:
            min_x = min(self.bounding_box_upper['min'][0], self.bounding_box_lower['min'][0])
            min_y = min(self.bounding_box_upper['min'][1], self.bounding_box_lower['min'][1])
            min_z = min(self.bounding_box_upper['min'][2], self.bounding_box_lower['min'][2])
            max_x = max(self.bounding_box_upper['max'][0], self.bounding_box_lower['max'][0])
            max_y = max(self.bounding_box_upper['max'][1], self.bounding_box_lower['max'][1])
            max_z = max(self.bounding_box_upper['max'][2], self.bounding_box_lower['max'][2])
            self.common_bbox = {
                'min': (min_x, min_y, min_z),
                'max': (max_x, max_y, max_z)
            }

    def _get_section_points(self, shape, plane, main_axis):
        if shape is None:
            return None
        try:
            section = BRepAlgoAPI_Section(shape, BRepBuilderAPI_MakeFace(plane).Face())
            section.Build()
            section_shape = section.Shape()
            if section_shape.IsNull():
                return None
            all_points = []
            edge_explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                curve = BRepAdaptor_Curve(edge)
                first = curve.FirstParameter()
                last = curve.LastParameter()
                num_points = 100
                step = (last - first) / (num_points - 1)
                for i in range(num_points):
                    param = first + i * step
                    pnt = curve.Value(param)
                    if main_axis == 0:
                        all_points.append([pnt.Y(), pnt.Z()])
                    elif main_axis == 1:
                        all_points.append([pnt.X(), pnt.Z()])
                    else:
                        all_points.append([pnt.X(), pnt.Y()])
                edge_explorer.Next()
            if len(all_points) < 3:
                return None
            points_array = np.array(all_points)
            unique = np.unique(np.round(points_array, decimals=4), axis=0)
            return unique if len(unique) >= 3 else points_array[:100]
        except Exception:
            return None

    def slice_wing_two_files(self, num_sections=10, use_manual=False, manual_z_coords=None, approx_points=200):
        """Нарезает сечения по Z координатам и сразу аппроксимирует их"""
        if not self.upper_shape or not self.lower_shape:
            raise Exception("Обе поверхности должны быть загружены")
        self.sections_data = []
        if use_manual and manual_z_coords:
            positions = manual_z_coords
            num_sections = len(positions)
        else:
            min_z = self.common_bbox['min'][2]
            max_z = self.common_bbox['max'][2]
            positions = []
            for i in range(num_sections):
                t = i / (num_sections - 1) if num_sections > 1 else 0.5
                positions.append(min_z + t * (max_z - min_z))
        for i, position in enumerate(positions):
            plane = gp_Pln(gp_Pnt(0, 0, position), gp_Dir(0, 0, 1))
            main_axis = 2
            upper_raw = self._get_section_points(self.upper_shape, plane, main_axis)
            lower_raw = self._get_section_points(self.lower_shape, plane, main_axis)
            log_msg = f"Сечение {i+1}/{num_sections} (Z={position:.2f}): "
            if upper_raw is None or lower_raw is None or len(upper_raw) < 3 or len(lower_raw) < 3:
                log_msg += "недостаточно данных"
                self.sections_data.append((None, [], False, None, None, position, None, None))
                yield log_msg
                continue
            le_idx_upper = np.argmin(upper_raw[:, 0])
            le_idx_lower = np.argmin(lower_raw[:, 0])
            le_point = upper_raw[le_idx_upper] if upper_raw[le_idx_upper, 0] < lower_raw[le_idx_lower, 0] else lower_raw[le_idx_lower]
            upper_sorted = upper_raw[np.argsort(upper_raw[:, 0])]
            lower_sorted = lower_raw[np.argsort(lower_raw[:, 0])]
            if not np.any(np.all(np.isclose(upper_sorted, le_point, atol=1e-5), axis=1)):
                upper_sorted = np.vstack(([le_point], upper_sorted))
            if not np.any(np.all(np.isclose(lower_sorted, le_point, atol=1e-5), axis=1)):
                lower_sorted = np.vstack(([le_point], lower_sorted))
            upper_sorted = upper_sorted[np.argsort(upper_sorted[:, 0])]
            lower_sorted = lower_sorted[np.argsort(lower_sorted[:, 0])]
            # Удаляем дубликаты
            upper_unique = []
            for point in upper_sorted:
                if not any(np.all(np.isclose(point, up, atol=1e-5)) for up in upper_unique):
                    upper_unique.append(point)
            upper_sorted = np.array(upper_unique)
            lower_unique = []
            for point in lower_sorted:
                if not any(np.all(np.isclose(point, low, atol=1e-5)) for low in lower_unique):
                    lower_unique.append(point)
            lower_sorted = np.array(lower_unique)
            # Аппроксимация
            upper_approx = None
            lower_approx = None
            try:
                upper_approx, lower_approx = SectionTools.auto_approximate_section(
                    upper_sorted, lower_sorted, approx_points, degree=5
                )
                if upper_approx is not None and lower_approx is not None:
                    log_msg += f"✓ аппроксимировано ({len(upper_approx)} верх, {len(lower_approx)} низ)"
                else:
                    log_msg += f"⚠ аппроксимация не удалась"
            except Exception as e:
                log_msg += f"⚠ ошибка аппроксимации: {str(e)[:50]}"
            # Построение контура
            lower_for_contour = lower_sorted[::-1]
            all_points = np.vstack((upper_sorted, lower_for_contour))
            n = len(all_points)
            connections = [[] for _ in range(n)]
            for j in range(n - 1):
                connections[j].append(j + 1)
                connections[j + 1].append(j)
            connections[0].append(n - 1)
            connections[n - 1].append(0)
            is_closed = True
            for j in range(n):
                if len(connections[j]) != 2:
                    is_closed = False
                    break
            self.sections_data.append((
                all_points, connections, is_closed,
                upper_sorted, lower_sorted, position,
                upper_approx, lower_approx
            ))
            yield log_msg
        valid = sum(1 for s in self.sections_data if s[0] is not None and len(s[0]) >= 3)
        yield f"Обработка завершена. Успешно: {valid}/{num_sections}"

    def get_section_data(self, index):
        """Возвращает данные сечения для MatplotlibSectionViewer."""
        if 0 <= index < len(self.sections_data):
            data = self.sections_data[index]
            return data[0], data[1], data[2]
        return None, [], False


class TwoFilesProcessingThread(QThread):
    finished = Signal(object)
    progress = Signal(int)
    log = Signal(str)

    def __init__(self, processor, num_sections, approx_points, use_manual=False, manual_z_coords=None):
        super().__init__()
        self.processor = processor
        self.num_sections = num_sections
        self.approx_points = approx_points
        self.use_manual = use_manual
        self.manual_z_coords = manual_z_coords or []

    def run(self):
        if self.use_manual:
            self.log.emit(f"Начало обработки с ручными Z координатами: {self.manual_z_coords}")
        else:
            self.log.emit(f"Начало автоматической обработки с {self.num_sections} сечениями")
        self.log.emit(f"Количество опорных точек: {self.approx_points}")
        section_count = 0
        for log_msg in self.processor.slice_wing_two_files(
            self.num_sections,
            use_manual=self.use_manual,
            manual_z_coords=self.manual_z_coords,
            approx_points=self.approx_points
        ):
            section_count += 1
            self.log.emit(log_msg)
            self.progress.emit(int(5 + 85 * (section_count / self.num_sections)))
        self.finished.emit(self.processor.sections_data)


class TwoFilesMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = TwoFilesSliceProcessor()
        self.upper_file = None
        self.lower_file = None
        self.sections = []
        self.total_sections = 0
        self.current_approx_points = 200
        self.manual_z_coords = []
        self.upper_shape = None
        self.lower_shape = None
        self.init_ui()

    def init_ui(self):
        title = "Анализ крыла по двум поверхностям"
        if OCC_SUPPORT:
            title += " (STEP поддержка)"
        else:
            title += " (⚠ OCC не доступен)"
        self.setWindowTitle(title)
        self.resize(900, 800)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        files_group = QGroupBox("Загрузка двух поверхностей")
        files_layout = QVBoxLayout()
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(QLabel("Верхняя поверхность:"))
        self.upper_label = QLabel("Файл не выбран")
        self.upper_label.setStyleSheet("color: #999;")
        upper_layout.addWidget(self.upper_label, 1)
        self.upper_btn = QPushButton("Загрузить верх")
        self.upper_btn.clicked.connect(self.load_upper_file)
        self.upper_btn.setEnabled(OCC_SUPPORT)
        upper_layout.addWidget(self.upper_btn)
        files_layout.addLayout(upper_layout)

        lower_layout = QHBoxLayout()
        lower_layout.addWidget(QLabel("Нижняя поверхность:"))
        self.lower_label = QLabel("Файл не выбран")
        self.lower_label.setStyleSheet("color: #999;")
        lower_layout.addWidget(self.lower_label, 1)
        self.lower_btn = QPushButton("Загрузить низ")
        self.lower_btn.clicked.connect(self.load_lower_file)
        self.lower_btn.setEnabled(OCC_SUPPORT)
        lower_layout.addWidget(self.lower_btn)
        files_layout.addLayout(lower_layout)
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        self.bbox_info = QLabel("Bounding box появится после загрузки обоих файлов")
        self.bbox_info.setStyleSheet("background-color: black; padding: 5px;")
        layout.addWidget(self.bbox_info)

        params_group = QGroupBox("Параметры нарезки")
        params_layout = QGridLayout()
        self.slice_method_group = QButtonGroup()
        self.auto_radio = QRadioButton("Автоматически (равномерно вдоль плоскости XOY)")
        self.auto_radio.setChecked(True)
        self.manual_radio = QRadioButton("Вручную (указать Z координаты)")
        self.slice_method_group.addButton(self.auto_radio)
        self.slice_method_group.addButton(self.manual_radio)
        self.auto_radio.toggled.connect(self.on_slice_method_changed)
        self.manual_radio.toggled.connect(self.on_slice_method_changed)
        method_layout = QHBoxLayout()
        method_layout.addWidget(self.auto_radio)
        method_layout.addWidget(self.manual_radio)
        method_layout.addStretch()
        params_layout.addLayout(method_layout, 0, 0, 1, 4)

        params_layout.addWidget(QLabel("Количество сечений:"), 1, 0)
        self.section_spinbox = QSpinBox()
        self.section_spinbox.setRange(2, 30)
        self.section_spinbox.setValue(10)
        params_layout.addWidget(self.section_spinbox, 1, 1)

        self.manual_btn = QPushButton("Задать Z координаты...")
        self.manual_btn.clicked.connect(self.show_manual_sections_dialog)
        self.manual_btn.setEnabled(False)
        params_layout.addWidget(self.manual_btn, 1, 2, 1, 2)

        self.z_coords_label = QLabel("")
        self.z_coords_label.setStyleSheet("color: #666; font-size: 9pt;")
        self.z_coords_label.setWordWrap(True)
        params_layout.addWidget(self.z_coords_label, 2, 0, 1, 4)

        params_layout.addWidget(QLabel("Опорных точек:"), 3, 0)
        self.approx_spinbox = QSpinBox()
        self.approx_spinbox.setRange(10, 1000)
        self.approx_spinbox.setValue(200)
        self.approx_spinbox.setSingleStep(10)
        params_layout.addWidget(self.approx_spinbox, 3, 1)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        buttons_layout = QHBoxLayout()
        self.process_btn = QPushButton("Нарезать сечения")
        self.process_btn.clicked.connect(self.process_model)
        self.process_btn.setEnabled(False)
        buttons_layout.addWidget(self.process_btn)
        self.view_btn = QPushButton("Просмотреть сечения")
        self.view_btn.clicked.connect(self.view_sections)
        self.view_btn.setEnabled(False)
        buttons_layout.addWidget(self.view_btn)
        self.view_3d_btn = QPushButton("3D модель крыла")
        self.view_3d_btn.clicked.connect(self.view_3d_model)
        self.view_3d_btn.setEnabled(False)
        buttons_layout.addWidget(self.view_3d_btn)
        self.export_056_btn = QPushButton("Экспорт .056")
        self.export_056_btn.clicked.connect(self.export_056)
        self.export_056_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_056_btn)
        layout.addLayout(buttons_layout)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        central.setLayout(layout)

        if not OCC_SUPPORT:
            self.log_text.append("⚠ ВНИМАНИЕ: pythonocc-core не доступен!")
        if PYVISTA_AVAILABLE:
            self.log_text.append("✓ PyVista доступен для улучшенной 3D визуализации")
        else:
            self.log_text.append("⚠ PyVista не установлен. Установите: pip install pyvista")

    def on_slice_method_changed(self):
        is_manual = self.manual_radio.isChecked()
        self.section_spinbox.setEnabled(not is_manual)
        self.manual_btn.setEnabled(is_manual)

    def show_manual_sections_dialog(self):
        if not self.processor.common_bbox:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите оба файла для определения диапазона Z")
            return
        min_z = self.processor.common_bbox['min'][2]
        max_z = self.processor.common_bbox['max'][2]
        dialog = ManualSectionsDialog(self, min_z, max_z, self.manual_z_coords)
        if dialog.exec_() == QDialog.Accepted:
            self.manual_z_coords = dialog.get_coordinates()
            if self.manual_z_coords:
                if len(self.manual_z_coords) <= 5:
                    coords_str = ", ".join([f"{z:.1f}" for z in self.manual_z_coords])
                else:
                    coords_str = f"{len(self.manual_z_coords)} сечений: " + \
                                 f"{self.manual_z_coords[0]:.1f}, {self.manual_z_coords[1]:.1f}, ..., " + \
                                 f"{self.manual_z_coords[-2]:.1f}, {self.manual_z_coords[-1]:.1f}"
                self.z_coords_label.setText(f"✓ Координаты: {coords_str}")
                self.z_coords_label.setStyleSheet("color: green; font-size: 9pt;")
            else:
                self.z_coords_label.setText("⚠ Координаты не заданы")
                self.z_coords_label.setStyleSheet("color: orange; font-size: 9pt;")

    def load_upper_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите STEP-файл верхней поверхности", "", "STEP (*.stp *.step)")
        if file_path:
            self.upper_file = file_path
            self.upper_label.setText(os.path.basename(file_path))
            self.upper_label.setStyleSheet("color: black; font-weight: bold;")
            if self.processor.load_upper_file(file_path):
                self.upper_shape = self.processor.upper_shape
                self.log_text.append(f"✓ Верхняя поверхность загружена: {os.path.basename(file_path)}")
            else:
                self.log_text.append(f"✗ Ошибка загрузки верхней поверхности")
            self.check_both_files_loaded()
            self.update_bbox_info()

    def load_lower_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите STEP-файл нижней поверхности", "", "STEP (*.stp *.step)")
        if file_path:
            self.lower_file = file_path
            self.lower_label.setText(os.path.basename(file_path))
            self.lower_label.setStyleSheet("color: black; font-weight: bold;")
            if self.processor.load_lower_file(file_path):
                self.lower_shape = self.processor.lower_shape
                self.log_text.append(f"✓ Нижняя поверхность загружена: {os.path.basename(file_path)}")
            else:
                self.log_text.append(f"✗ Ошибка загрузки нижней поверхности")
            self.check_both_files_loaded()
            self.update_bbox_info()

    def check_both_files_loaded(self):
        ready = bool(self.upper_file and self.lower_file and OCC_SUPPORT)
        self.process_btn.setEnabled(ready)

    def update_bbox_info(self):
        if self.processor.common_bbox:
            common = self.processor.common_bbox
            info = f"Общий диапазон: X:{common['max'][0]-common['min'][0]:.1f} " \
                   f"Y:{common['max'][1]-common['min'][1]:.1f} " \
                   f"Z:{common['max'][2]-common['min'][2]:.1f}"
            self.bbox_info.setText(info)

    def process_model(self):
        if not self.upper_file or not self.lower_file:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите оба файла")
            return
        self.process_btn.setEnabled(False)
        self.view_btn.setEnabled(False)
        self.view_3d_btn.setEnabled(False)
        self.export_056_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_text.clear()
        self.current_approx_points = self.approx_spinbox.value()
        use_manual = self.manual_radio.isChecked() and self.manual_z_coords
        if use_manual:
            num_sections = len(self.manual_z_coords)
        else:
            num_sections = self.section_spinbox.value()
        self.thread = TwoFilesProcessingThread(
            self.processor, num_sections, self.current_approx_points, use_manual, self.manual_z_coords
        )
        self.thread.finished.connect(self.on_finished)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.log.connect(self.log_text.append)
        self.thread.start()

    def on_finished(self, sections):
        self.process_btn.setEnabled(True)
        self.progress.setVisible(False)
        if sections is not None:
            self.sections = sections
            self.total_sections = len(sections)
            valid = sum(1 for s in sections if s[0] is not None and len(s[0]) >= 3)
            approx_valid = sum(1 for s in sections if len(s) >= 8 and s[6] is not None)
            self.log_text.append(f"\n✓ Итог: успешно обработано {valid}/{self.total_sections} сечений")
            self.log_text.append(f"✓ Аппроксимировано: {approx_valid}/{valid} сечений")
            self.view_btn.setEnabled(valid > 0)
            self.view_3d_btn.setEnabled(valid > 0)
            self.export_056_btn.setEnabled(valid > 0)
            if valid > 0:
                QMessageBox.information(self, "Успех",
                    f"Создано {valid} сечений!\nИз них аппроксимировано: {approx_valid}")
        else:
            self.log_text.append("✗ Ошибка обработки")

    def view_sections(self):
        if self.total_sections > 0:
            for i in range(self.total_sections):
                data = self.processor.get_section_data(i)
                if data[0] is not None and len(data[0]) >= 3:
                    self.show_section(i)
                    break

    def show_section(self, index):
        if index < 0 or index >= len(self.sections):
            return
        data_full = self.sections[index]
        if len(data_full) < 6:
            return
        points, connections, is_closed, upper_points, lower_points, position = data_full[:6]
        if points is None or len(points) < 3:
            return
        viewer = MatplotlibSectionViewer(
            section_data=(points, connections, is_closed),
            section_number=index,
            total_sections=self.total_sections,
            cut_axis_info=f"{position:.2f}",
            upper_points=upper_points,
            lower_points=lower_points,
            parent=self
        )
        viewer.exec_()

    def view_3d_model(self):
        if not hasattr(self, 'sections') or not self.sections:
            return
        valid_sections = [s for s in self.sections if s[0] is not None and len(s[0]) >= 3]
        if len(valid_sections) < 2:
            QMessageBox.warning(self, "Ошибка", "Недостаточно сечений для 3D визуализации")
            return
        if PYVISTA_AVAILABLE:
            Wing3DVisualizerPyVista(valid_sections, self.upper_shape, self.lower_shape).visualize()
        else:
            Wing3DViewer(self.upper_shape, self.lower_shape, valid_sections, self).exec_()

    def export_056(self):
        if self.sections:
            ExportUtils.export_to_056(self.sections, parent_window=self)
        else:
            QMessageBox.warning(self, "Ошибка", "Нет данных сечений для экспорта")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TwoFilesMainWindow()
    window.show()
    sys.exit(app.exec())