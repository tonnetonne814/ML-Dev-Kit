#!/usr/bin/env python3
"""PyTorch、torchvision、torchaudioのCUDAサポートを確認するスクリプト"""

import sys
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def check_cuda_support():
    """各ライブラリのCUDAサポート状況を確認"""
    print("=" * 60)
    print("CUDAサポート確認")
    print("=" * 60)
    
    # PyTorchの確認
    try:
        import torch
        print(f"\n[PyTorch]")
        print(f"  バージョン: {torch.__version__}")
        print(f"  CUDA利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDAバージョン: {torch.version.cuda}")
            print(f"  cuDNNバージョン: {torch.backends.cudnn.version()}")
            print(f"  GPU数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"      メモリ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("  CUDAは利用できません")
    except ImportError:
        print("\n[PyTorch] インストールされていません")
    
    # torchvisionの確認
    try:
        import torchvision
        print(f"\n[torchvision]")
        print(f"  バージョン: {torchvision.__version__}")
        if 'torch' in sys.modules and torch.cuda.is_available():
            print(f"  CUDAサポート: あり")
            # 簡単なテスト
            try:
                x = torch.randn(1, 3, 224, 224).cuda()
                print(f"  CUDA動作確認: 成功")
            except Exception as e:
                print(f"  CUDA動作確認: 失敗 ({e})")
        else:
            print(f"  CUDAサポート: なし（CUDAが利用できないため）")
    except ImportError:
        print("\n[torchvision] インストールされていません")
    
    # torchaudioの確認
    try:
        import torchaudio
        print(f"\n[torchaudio]")
        print(f"  バージョン: {torchaudio.__version__}")
        if 'torch' in sys.modules and torch.cuda.is_available():
            print(f"  CUDAサポート: あり")
            # 簡単なテスト
            try:
                x = torch.randn(1, 1, 16000).cuda()
                print(f"  CUDA動作確認: 成功")
            except Exception as e:
                print(f"  CUDA動作確認: 失敗 ({e})")
        else:
            print(f"  CUDAサポート: なし（CUDAが利用できないため）")
    except ImportError:
        print("\n[torchaudio] インストールされていません")
    
    print("\n" + "=" * 60)
    
    # 総合判定
    try:
        if 'torch' in sys.modules and torch.cuda.is_available():
            print("✓ CUDAは利用可能です")
            return 0
        else:
            print("✗ CUDAは利用できません")
            return 1
    except NameError:
        print("✗ PyTorchがインストールされていないため、CUDAの確認ができません")
        return 1

if __name__ == "__main__":
    sys.exit(check_cuda_support())

