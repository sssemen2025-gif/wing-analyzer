#!/usr/bin/env python3
"""
Wing Analyzer - запуск приложения
"""

import sys
from PySide6.QtWidgets import QApplication
from src.wing_analyzer_two_files import TwoFilesMainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TwoFilesMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()