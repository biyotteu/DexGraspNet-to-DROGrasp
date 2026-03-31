"""
Euler Angle 변환 검증 테스트
============================

DexGraspNet: transforms3d.euler.euler2mat(rx, ry, rz) → 'sxyz' (static/extrinsic XYZ)
DRO-Grasp:   scipy Rotation.from_euler('XYZ', [rx, ry, rz]) → intrinsic XYZ

핵심: extrinsic 'xyz'(소문자) ≡ intrinsic 'ZYX'(대문자 역순)
      이 둘은 다른 컨벤션이므로 rotation matrix를 중간 매개로 변환해야 함.

이 테스트는 두 가지를 검증합니다:
1. transforms3d sxyz → rotation matrix → scipy XYZ intrinsic 변환의 정확성
2. 변환 후 rotation matrix가 동일한지 확인
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Optional: transforms3d가 설치된 경우에만 직접 비교
try:
    import transforms3d
    HAS_TRANSFORMS3D = True
except ImportError:
    HAS_TRANSFORMS3D = False
    print("transforms3d not installed. Using scipy-only verification.")


def test_euler_convention_equivalence():
    """
    transforms3d.euler.euler2mat(rx, ry, rz) == scipy Rotation.from_euler('xyz', [rx, ry, rz])
    둘 다 extrinsic XYZ (static frame) 회전을 나타냄을 검증
    """
    print("Test 1: transforms3d sxyz ≡ scipy extrinsic xyz")
    print("-" * 50)

    np.random.seed(42)
    for i in range(10):
        angles = np.random.uniform(-np.pi, np.pi, 3)
        rx, ry, rz = angles

        # scipy extrinsic xyz (lowercase)
        R_scipy = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()

        if HAS_TRANSFORMS3D:
            # transforms3d sxyz (default)
            R_t3d = transforms3d.euler.euler2mat(rx, ry, rz, axes='sxyz')
            error = np.abs(R_scipy - R_t3d).max()
            status = "PASS" if error < 1e-10 else "FAIL"
            print(f"  Sample {i+1}: error={error:.2e} [{status}]")
        else:
            print(f"  Sample {i+1}: (skipped - transforms3d not available)")

    print()


def test_conversion_roundtrip():
    """
    DexGraspNet euler → matrix → DRO-Grasp euler → matrix 변환의 정확성 검증
    """
    print("Test 2: Full conversion round-trip")
    print("-" * 50)

    np.random.seed(42)
    max_error = 0

    for i in range(100):
        # Random DexGraspNet euler angles
        rx, ry, rz = np.random.uniform(-np.pi, np.pi, 3)

        # Step 1: DexGraspNet euler → rotation matrix
        # (transforms3d sxyz ≡ scipy extrinsic xyz)
        R_original = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()

        # Step 2: matrix → DRO-Grasp euler (intrinsic XYZ)
        euler_dro = Rotation.from_matrix(R_original).as_euler('XYZ')

        # Step 3: DRO-Grasp euler → matrix (verify)
        R_reconstructed = Rotation.from_euler('XYZ', euler_dro).as_matrix()

        error = np.abs(R_original - R_reconstructed).max()
        max_error = max(max_error, error)

        if error > 1e-10:
            print(f"  Sample {i+1}: FAIL (error={error:.2e})")
            print(f"    DexGraspNet: ({rx:.4f}, {ry:.4f}, {rz:.4f})")
            print(f"    DRO-Grasp:   ({euler_dro[0]:.4f}, {euler_dro[1]:.4f}, {euler_dro[2]:.4f})")

    status = "PASS" if max_error < 1e-10 else "FAIL"
    print(f"  Max error across 100 samples: {max_error:.2e} [{status}]")
    print()


def test_specific_cases():
    """
    특정 케이스에 대한 검증 (단위 변환, 90도 회전 등)
    """
    print("Test 3: Specific rotation cases")
    print("-" * 50)

    test_cases = [
        ("Identity", (0, 0, 0)),
        ("90° around X", (np.pi/2, 0, 0)),
        ("90° around Y", (0, np.pi/2, 0)),
        ("90° around Z", (0, 0, np.pi/2)),
        ("-45° around all", (-np.pi/4, -np.pi/4, -np.pi/4)),
        ("Mixed", (0.5, -1.2, 2.1)),
    ]

    for name, (rx, ry, rz) in test_cases:
        # DexGraspNet → matrix
        R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()

        # matrix → DRO-Grasp euler
        euler_dro = Rotation.from_matrix(R).as_euler('XYZ')

        # DRO-Grasp euler → matrix (verify)
        R_check = Rotation.from_euler('XYZ', euler_dro).as_matrix()

        error = np.abs(R - R_check).max()
        status = "PASS" if error < 1e-10 else "FAIL"

        print(f"  {name:20s}: DexGN=({rx:+.3f}, {ry:+.3f}, {rz:+.3f}) "
              f"→ DRO=({euler_dro[0]:+.3f}, {euler_dro[1]:+.3f}, {euler_dro[2]:+.3f}) "
              f"err={error:.1e} [{status}]")

    print()


def test_extrinsic_vs_intrinsic_difference():
    """
    extrinsic xyz와 intrinsic XYZ가 실제로 다른 결과를 내는지 확인.
    이 테스트가 통과하면 변환이 필요함을 확인.
    """
    print("Test 4: Extrinsic xyz ≠ Intrinsic XYZ (confirms conversion is needed)")
    print("-" * 50)

    np.random.seed(42)
    different_count = 0

    for i in range(100):
        angles = np.random.uniform(-np.pi, np.pi, 3)

        R_extrinsic = Rotation.from_euler('xyz', angles).as_matrix()
        R_intrinsic = Rotation.from_euler('XYZ', angles).as_matrix()

        if not np.allclose(R_extrinsic, R_intrinsic, atol=1e-10):
            different_count += 1

    print(f"  {different_count}/100 random angle sets produce different matrices")
    print(f"  {'PASS' if different_count > 50 else 'FAIL'}: "
          f"Conversion is {'needed' if different_count > 0 else 'NOT needed'}")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Euler Angle Conversion Verification Tests")
    print("=" * 60)
    print()

    test_euler_convention_equivalence()
    test_conversion_roundtrip()
    test_specific_cases()
    test_extrinsic_vs_intrinsic_difference()

    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)
