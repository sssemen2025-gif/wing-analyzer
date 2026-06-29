
---

## 🔧 Установка и запуск

### Требования

- Python 3.9 или новее
- Git (для клонирования)
- Visual C++ Redistributable (для pythonocc-core на Windows)

### Установка

```bash
# Клонируем репозиторий
git clone https://github.com/sssemen2025-gif/wing-analyzer.git
cd wing-analyzer

# Создаём виртуальное окружение
python -m venv venv

# Активируем
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt
