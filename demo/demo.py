from src.risk.fusion import fuse_risk
from src.risk.mocks import SCENARIOS


EXPECTED = {
    "1_real_normal": "LOW",
    "2_synthetic_normal": "LOW",
    "3_real_suspicious": "HIGH",
    "4_synthetic_suspicious": "HIGH",
    "5_hard_negative": "LOW",
}


def main():
    print()
    print("SCENARIO                    SCORE  LEVEL     ACTION    EXPECTED  MATCH")
    print("-" * 80)

    all_match = True

    for name, payload in SCENARIOS.items():
        result = fuse_risk(payload)
        expected = EXPECTED[name]

        match = result.risk_level == expected
        all_match = all_match and match

        print(
            f"{name:<27}"
            f"{result.final_risk_score:<7.3f}"
            f"{result.risk_level:<10}"
            f"{result.action:<10}"
            f"{expected:<10}"
            f"{'OK' if match else 'FAIL'}"
        )

        for reason in result.reasons:
            print(f"    - {reason}")

        for warning in result.warnings:
            print(f"    ! {warning}")

        print()

    if all_match:
        print("ALL SCENARIOS MATCH EXPECTED BEHAVIOR")
    else:
        print("SOME SCENARIOS DO NOT MATCH EXPECTED BEHAVIOR")


if __name__ == "__main__":
    main()