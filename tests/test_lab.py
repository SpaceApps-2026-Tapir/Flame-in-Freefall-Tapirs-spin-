"""Offline checks for the visual lab engine and the agent's tool loop (no network)."""
import json
import unittest

import lab_agent
import lab_engine as L


class EngineTests(unittest.TestCase):
    def test_every_analysis_runs_on_every_dataset(self):
        for ds in L.DATASETS:
            spec = L.default_spec(ds)
            for name in L.ANALYSES:
                if name == "nearest_to_scenario" and L.DATASETS[ds]["o2"] is None:
                    with self.assertRaises(ValueError):
                        L.run_analysis(spec, name)
                    continue
                new, res = L.run_analysis(spec, name)
                self.assertTrue(res["summary"])
                L.make_figure(new)

    def test_design_matrix_weights_and_filters(self):
        spec = L.apply_tool(L.default_spec("all"), "set_design_matrix", {
            "only": ["oxygen_pct", "pressure_kpa"], "variables": [{"variable": "oxygen_pct", "weight": 3}],
            "filters": {"experiment": ["BASS", "BASS-II"]}})
        X, names = L.design_matrix(spec)
        self.assertEqual(names, ["oxygen_pct", "pressure_kpa"])
        self.assertAlmostEqual(X[:, 0].std() / X[:, 1].std(), 3, places=1)
        self.assertEqual(set(L.filtered(spec).experiment), {"BASS", "BASS-II"})

    def test_bad_input_is_rejected(self):
        spec = L.default_spec("all")
        for tool, args in [("set_design_matrix", {"variables": [{"variable": "nope"}]}),
                           ("set_view", {"method": "magic"}), ("set_view", {"x": "experiment"})]:
            with self.assertRaises(ValueError):
                L.apply_tool(spec, tool, args)

    def test_camera_moves_bump_version(self):
        spec = L.apply_tool(L.default_spec("all"), "set_view", {"rotate_degrees": 90})
        self.assertEqual(spec["camera"]["azimuth"], 135)
        self.assertEqual(spec["version"], 1)


class AgentTests(unittest.TestCase):
    def test_tool_loop_applies_actions_and_reports_errors(self):
        script = [
            {"tool_calls": [
                {"id": "1", "function": {"name": "set_design_matrix",
                                         "arguments": json.dumps({"only": ["oxygen_pct", "pressure_kpa"]})}},
                {"id": "2", "function": {"name": "set_view", "arguments": json.dumps({"method": "pca", "rotate_degrees": 30})}},
                {"id": "3", "function": {"name": "set_view", "arguments": json.dumps({"x": "not_a_column"})}},
                {"id": "4", "function": {"name": "run_analysis", "arguments": json.dumps({"name": "kmeans", "params": {"k": 3}})}},
            ]},
            {"content": "Clustered on oxygen and pressure."},
        ]
        seen = []

        def fake(messages):
            seen.append(messages[-1])
            return script[len(seen) - 1]

        spec, reply, actions, results = lab_agent.run(L.default_spec("all"), [], "cluster by O2 and pressure", call=fake)
        self.assertEqual(reply, "Clustered on oxygen and pressure.")
        self.assertEqual([a["ok"] for a in actions], [True, True, False, True])
        self.assertEqual(spec["method"], "pca")
        self.assertEqual(spec["color"], "cluster")
        self.assertEqual(results[0]["name"], "kmeans")
        self.assertEqual(seen[-1]["role"], "tool")  # tool results were fed back before the final answer

    def test_missing_key_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            lab_agent.run(L.default_spec("all"), [], "hi", api_key="", model="")


if __name__ == "__main__":
    unittest.main()


class GuideTests(unittest.TestCase):
    def test_routes(self):
        from guide import BRANCHES, route
        self.assertEqual(route("Can smoke tell which material is burning?")["data"], "same")
        r = route("Is Nomex safe in a lunar rover?")
        self.assertEqual((r["page"], r["pillar"]), ("ask", "prevention"))
        self.assertEqual(route("Why do flames go out?")["page"], "ask")
        for b in BRANCHES:
            if b["params"]["page"] == "visual":
                self.assertIn(b["params"]["chart"], L.CHARTS_FOR[b["params"]["data"]])
