from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    """MNISTデータセット用の`LightningDataModule`。

    MNISTデータベースの手書き数字には、60,000例のトレーニングセットと10,000例のテストセットがあります。
    これはNISTから入手可能なより大きなセットのサブセットです。数字はサイズが正規化され、固定サイズの
    画像の中心に配置されています。NISTの元の白黒画像は、アスペクト比を保ちながら20x20ピクセルのボックスに
    収まるようにサイズが正規化されました。正規化アルゴリズムで使用されたアンチエイリアシング技術の結果として、
    結果画像にはグレーレベルが含まれています。ピクセルの重心を計算し、この点が28x28フィールドの中心に
    位置するように画像を変換することで、画像は28x28画像の中心に配置されました。

    `LightningDataModule`は7つの主要なメソッドを実装します：

    ```python
        def prepare_data(self):
        # 1つのGPU/TPUで行うこと（DDPの全てのGPU/TPUで行うわけではない）。
        # データのダウンロード、前処理、分割、ディスクへの保存など...

        def setup(self, stage):
        # DDPの全てのプロセスで行うこと。
        # データの読み込み、変数の設定など...

        def train_dataloader(self):
        # トレーニングデータローダーを返す

        def val_dataloader(self):
        # 検証データローダーを返す

        def test_dataloader(self):
        # テストデータローダーを返す

        def predict_dataloader(self):
        # 予測データローダーを返す

        def teardown(self, stage):
        # DDPの全てのプロセスで呼び出される。
        # fitまたはtestの後のクリーンアップ。
    ```

    これにより、ダウンロード方法、分割方法、変換方法、データの処理方法を説明せずに、
    完全なデータセットを共有することができます。

    ドキュメントを読む：
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """MNISTDataModuleを初期化します。

        :param data_dir: データディレクトリ。デフォルトは`"data/"`。
        :param train_val_test_split: トレーニング、検証、テストの分割。デフォルトは`(55_000, 5_000, 10_000)`。
        :param batch_size: バッチサイズ。デフォルトは`64`。
        :param num_workers: ワーカーの数。デフォルトは`0`。
        :param pin_memory: メモリをピンするかどうか。デフォルトは`False`。
        """
        super().__init__()

        # この行により、'self.hparams'属性で初期化パラメータにアクセスできます
        # また、初期化パラメータがckptに保存されることを保証します
        self.save_hyperparameters(logger=False)

        # データ変換
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """クラスの数を取得します。

        :return: MNISTクラスの数（10）。
        """
        return 10

    def prepare_data(self) -> None:
        """必要に応じてデータをダウンロードします。Lightningは`self.prepare_data()`がCPU上の単一プロセス内でのみ
        呼び出されることを保証するため、ダウンロードロジックを安全に追加できます。マルチノードトレーニングの場合、
        このフックの実行は`self.prepare_data_per_node()`に依存します。

        状態を割り当てるために使用しないでください（self.x = y）。
        """
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """データを読み込みます。変数を設定します：`self.data_train`、`self.data_val`、`self.data_test`。

        このメソッドは、Lightningによって`trainer.fit()`、`trainer.validate()`、`trainer.test()`、
        `trainer.predict()`の前に呼び出されるため、ランダム分割などを2回実行しないように注意してください！
        また、`self.prepare_data()`の後に呼び出され、その間にバリアがあり、データが準備され使用可能になると、
        すべてのプロセスが`self.setup()`に進むことが保証されます。

        :param stage: セットアップするステージ。`"fit"`、`"validate"`、`"test"`、または`"predict"`のいずれか。
            デフォルトは``None``。
        """
        # バッチサイズをデバイス数で割ります。
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"バッチサイズ（{self.hparams.batch_size}）がデバイス数（{self.trainer.world_size}）で割り切れません。"
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # まだロードされていない場合にのみデータセットを読み込んで分割します
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """トレーニングデータローダーを作成して返します。

        :return: トレーニングデータローダー。
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """検証データローダーを作成して返します。

        :return: 検証データローダー。
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """テストデータローダーを作成して返します。

        :return: テストデータローダー。
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightningフックで、`trainer.fit()`、`trainer.validate()`、`trainer.test()`、
        `trainer.predict()`の後のクリーンアップを行います。

        :param stage: 解体されるステージ。`"fit"`、`"validate"`、`"test"`、または`"predict"`のいずれか。
            デフォルトは``None``。
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """チェックポイントを保存するときに呼び出されます。データモジュールの状態を生成して保存するために実装します。

        :return: 保存したいデータモジュールの状態を含む辞書。
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """チェックポイントを読み込むときに呼び出されます。データモジュールの`state_dict()`によって返された
        データモジュールの状態を再読み込みするために実装します。

        :param state_dict: `self.state_dict()`によって返されたデータモジュールの状態。
        """
        pass


if __name__ == "__main__":
    _ = MNISTDataModule()