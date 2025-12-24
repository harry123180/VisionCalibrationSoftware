# 貢獻指南

感謝您有興趣為此專案做出貢獻！

[English](#english) | 繁體中文

## 如何貢獻

### 回報錯誤

1. 使用 [Issue 模板](https://github.com/TSIC-LAB/VisionCalibrationSoftware/issues/new?template=bug_report.md) 提交錯誤回報
2. 清楚描述問題和重現步驟
3. 附上相關的螢幕截圖或錯誤訊息

### 提出新功能

1. 使用 [功能建議模板](https://github.com/TSIC-LAB/VisionCalibrationSoftware/issues/new?template=feature_request.md)
2. 說明功能的用途和預期行為
3. 如果可能，提供實現思路

### 提交程式碼

1. Fork 此專案
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m '新增某個功能'`)
4. 推送至分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 開發環境設置

```bash
# Clone 專案
git clone https://github.com/TSIC-LAB/VisionCalibrationSoftware.git
cd VisionCalibrationSoftware

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安裝開發依賴
pip install -e ".[dev]"
```

## 程式碼規範

- 使用 `ruff` 進行程式碼風格檢查
- 使用 `mypy` 進行型別檢查
- 使用 `pytest` 進行測試
- 遵循 PEP 8 風格指南
- 使用有意義的變數和函數命名
- 在複雜邏輯處添加註解

## 測試

```bash
# 執行所有測試
pytest

# 執行特定測試
pytest tests/test_intrinsic.py

# 執行測試並顯示覆蓋率
pytest --cov=vision_calib
```

## 提交訊息規範

使用清楚的提交訊息：

- `feat:` 新功能
- `fix:` 錯誤修復
- `docs:` 文件更新
- `style:` 程式碼格式調整
- `refactor:` 程式碼重構
- `test:` 測試相關
- `chore:` 其他維護工作

範例：`feat: 新增 IPPE_SQUARE 算法支援`

---

<a name="english"></a>
# Contributing Guide (English)

Thank you for your interest in contributing!

## How to Contribute

### Reporting Bugs

1. Use the [bug report template](https://github.com/TSIC-LAB/VisionCalibrationSoftware/issues/new?template=bug_report.md)
2. Clearly describe the problem and steps to reproduce
3. Include relevant screenshots or error messages

### Suggesting Features

1. Use the [feature request template](https://github.com/TSIC-LAB/VisionCalibrationSoftware/issues/new?template=feature_request.md)
2. Explain the purpose and expected behavior
3. Provide implementation ideas if possible

### Submitting Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add some feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code Standards

- Use `ruff` for linting
- Use `mypy` for type checking
- Use `pytest` for testing
- Follow PEP 8 style guide
