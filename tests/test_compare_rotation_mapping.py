import unittest

from core.processing import (
    classify_compare_channel, coherent_compare_auto_assignment,
    group_compare_sources, infer_compare_angle_references,
    parse_compare_in_out_angles,
)


class RotationMappingTests(unittest.TestCase):
    def test_rot_in_out_tokens_are_semantic_and_case_insensitive(self):
        name = "YZ303_RotOut45deg_Stage800.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (None, 45.0))
        self.assertEqual(
            parse_compare_in_out_angles(name, rot1_is_output=True),
            (None, 45.0),
        )
        combo = "YZ303_RotIn10deg_RoToUt90DEG_Stage800.csv"
        self.assertEqual(parse_compare_in_out_angles(combo), (10.0, 90.0))
        self.assertEqual(
            parse_compare_in_out_angles(combo, rot1_is_output=True),
            (10.0, 90.0),
        )

    def test_rot1_rot2_tokens_follow_selected_mapping(self):
        name = "scan_Rot1_10deg_Rot2_90deg.csv"
        self.assertEqual(parse_compare_in_out_angles(name), (10.0, 90.0))
        self.assertEqual(
            parse_compare_in_out_angles(name, rot1_is_output=True),
            (90.0, 10.0),
        )

    def test_rot_in_out_angles_share_group_context_after_angle_stripping(self):
        files = [
            "YZ303_pe2_3.6KPL_730nm117.545uW_940nmc_1sx1_RotOut45deg_Stage800_TG-1.1BG=25.2.csv",
            "YZ303_pe2_3.6KPL_730nm117.545uW_940nmc_1sx1_RotOut90deg_Stage800_TG-1.1BG=25.2.csv",
        ]
        groups = group_compare_sources(
            files,
            in_k_angle=0.0,
            in_kp_angle=45.0,
            out_k_angle=45.0,
            out_kp_angle=90.0,
            tolerance=2.0,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].mapping, {"KK": files[0], "KKp": files[1]})

    def test_conflicting_semantic_tokens_are_unclassified(self):
        name = "scan_RotIn10deg_in20deg.csv"
        self.assertIsNone(
            classify_compare_channel(
                name,
                in_k_angle=0.0,
                in_kp_angle=90.0,
                out_k_angle=0.0,
                out_kp_angle=90.0,
                tolerance=2.0,
            )
        )

    def test_angle_conflict_does_not_fall_back_to_missing_arm_or_channel_token(self):
        conflict_with_output = "scan_RotIn10deg_in20deg_RotOut0deg.csv"
        conflict_with_channel = "scan_RotIn10deg_in20deg_KK.csv"
        for name in (conflict_with_output, conflict_with_channel):
            with self.subTest(name=name):
                self.assertIsNone(
                    classify_compare_channel(
                        name,
                        in_k_angle=0.0,
                        in_kp_angle=90.0,
                        out_k_angle=0.0,
                        out_kp_angle=90.0,
                        tolerance=2.0,
                    )
                )

    def test_unambiguous_missing_input_arm_keeps_default_k(self):
        self.assertEqual(
            classify_compare_channel(
                "scan_RotOut0deg.csv",
                in_k_angle=0.0,
                in_kp_angle=90.0,
                out_k_angle=0.0,
                out_kp_angle=90.0,
                tolerance=2.0,
            ),
            "KK",
        )

    def test_swapped_single_and_dual_rotation_channels(self):
        refs = dict(in_k_angle=10, in_kp_angle=55, out_k_angle=90,
                    out_kp_angle=45, tolerance=2, rot1_is_output=True)
        for name, expected in [
            ('scan_Rot190deg.csv', 'KK'),
            ('scan_Rot145deg.csv', 'KKp'),
            ('scan_Rot255deg.csv', 'KpK'),
            ('scan_Rot145deg_Rot255deg.csv', 'KpKp'),
            ('scan_KKp.csv', 'KKp'),
        ]:
            with self.subTest(name=name):
                self.assertEqual(classify_compare_channel(name, **refs), expected)

    def test_named_arms_remain_semantic_when_rotations_swap(self):
        for name, expected in [
            ('scan_in10deg_out90deg.csv', (10, 90)),
            ('scan_deg90.csv', (None, 90)),
            ('scan_Rot145deg_out90deg_Rot255deg.csv', (55, 90)),
        ]:
            self.assertEqual(parse_compare_in_out_angles(name, rot1_is_output=True), expected)

    def test_rot1_output_inference_and_group_assignment(self):
        files = ['sample_PL_Rot190deg.csv', 'sample_PL_Rot145deg.csv']
        inferred = infer_compare_angle_references(
            files, in_k_anchor=0, out_k_anchor=90, rot1_is_output=True)
        self.assertEqual((inferred.in_k, inferred.in_kp), (None, None))
        self.assertEqual((inferred.out_k, inferred.out_kp), (90, 45))
        refs = dict(in_k_angle=0, out_k_angle=90, out_kp_angle=45,
                    tolerance=2, rot1_is_output=True)
        self.assertEqual(group_compare_sources(files, **refs)[0].mapping,
                         {'KK': files[0], 'KKp': files[1]})
        self.assertEqual(coherent_compare_auto_assignment(files, **refs)[0],
                         {'KK': files[0], 'KKp': files[1]})
