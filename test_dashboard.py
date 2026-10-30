"""用 Streamlit AppTest 验证预测按钮和模式切换。

在项目目录运行：python -m unittest -v test_dashboard.py
不需要浏览器；不修改模型文件，也不使用网络。
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from streamlit.testing.v1 import AppTest

from model_adapter import predict_power
from wind_core import SOURCE_FEATURES


class DashboardInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assert_no_exception()

    def assert_no_exception(self) -> None:
        self.assertEqual([item.message for item in self.app.exception], [])

    def submit_prediction(self) -> dict | None:
        self.app.button(key="FormSubmitter:prediction_form-执行功率预测").click().run()
        self.assert_no_exception()
        return self.app.session_state["last_prediction"]

    def test_default_screen(self) -> None:
        """完整项目冷启动，默认展示 WT6，并等待用户提交。"""
        self.assertEqual(self.app.selectbox(key="turbine").value, "WT6")
        self.assertEqual(len(self.app.error), 0)
        self.assertIsNone(self.app.session_state["last_prediction"])
        self.assertTrue(self.app.button(key="batch_predict").disabled)
        self.assertEqual(len(self.app.number_input), 7)

    def test_prediction_button_calls_real_model_and_uses_changed_input(self) -> None:
        """一次提交即得到模型结果，修改风速后再次提交确实重新推理。"""
        result = self.submit_prediction()
        self.assertIsNotNone(result)
        self.assertTrue(math.isfinite(result["value"]))
        raw = pd.DataFrame([result["inputs"]], columns=SOURCE_FEATURES)
        expected = float(predict_power(raw, turbine="WT6")[0])
        self.assertAlmostEqual(result["value"], expected, places=9)
        previous = result["value"]

        self.app.number_input(key="feature_0").set_value(14.0)
        changed = self.submit_prediction()
        self.assertEqual(changed["inputs"]["V"], 14.0)
        self.assertNotAlmostEqual(changed["value"], previous, places=3)
        raw_changed = pd.DataFrame([changed["inputs"]], columns=SOURCE_FEATURES)
        self.assertAlmostEqual(changed["value"], float(predict_power(raw_changed, turbine="WT6")[0]), places=9)

    def test_turbine_switch_clears_result_then_predicts_wt5(self) -> None:
        """切换机组清空上次结果，新结果确实来自 WT5 模型。"""
        self.assertIsNotNone(self.submit_prediction())
        self.app.session_state["batch_result"] = {"sentinel": "previous turbine"}
        self.app.selectbox(key="turbine").select("WT5").run()
        self.assert_no_exception()
        self.assertIsNone(self.app.session_state["last_prediction"])
        self.assertIsNone(self.app.session_state["batch_result"])
        result = self.submit_prediction()
        self.assertEqual(result["turbine"], "WT5")
        self.assertEqual(result["context"], "WT5|demo|0")
        raw = pd.DataFrame([result["inputs"]], columns=SOURCE_FEATURES)
        self.assertAlmostEqual(result["value"], float(predict_power(raw, turbine="WT5")[0]), places=9)

    def test_missing_custom_model_is_explicit_and_never_falls_back(self) -> None:
        """自定义模型缺失时明确报错，不把内置模型结果冒充自定义结果。"""
        if (ROOT / "models" / "custom_model.joblib").is_file():
            self.skipTest("用户已放置自定义模型，不执行缺失文件场景。")
        self.assertIsNotNone(self.submit_prediction())
        self.app.selectbox(key="model_mode").select("自定义本地模型").run()
        self.assert_no_exception()
        self.assertIsNone(self.app.session_state["last_prediction"])
        self.assertIsNone(self.submit_prediction())
        self.assertTrue(any("未找到自定义模型" in item.value for item in self.app.error))
        self.app.selectbox(key="model_mode").select("内置示例模型").run()
        self.assert_no_exception()
        self.assertEqual(len(self.app.error), 0)
        self.assertIsNone(self.app.session_state["last_prediction"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
