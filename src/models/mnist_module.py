from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class MNISTLitModule(LightningModule):
    """MNIST分類のための`LightningModule`の例。

    `LightningModule`は8つの主要なメソッドを実装しています：

    ```python
    def __init__(self):
    # 初期化コードをここで定義します。

    def setup(self, stage):
    # 各ステージ（'fit'、'validate'、'test'、'predict'）の前に設定するもの。
    # このフックはDDPを使用する場合、すべてのプロセスで呼び出されます。

    def training_step(self, batch, batch_idx):
    # 完全なトレーニングステップ。

    def validation_step(self, batch, batch_idx):
    # 完全な検証ステップ。

    def test_step(self, batch, batch_idx):
    # 完全なテストステップ。

    def predict_step(self, batch, batch_idx):
    # 完全な予測ステップ。

    def configure_optimizers(self):
    # オプティマイザとLRスケジューラを定義および設定します。
    ```

    ドキュメント：
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """MNISTLitModuleを初期化します。

        :param net: トレーニングするモデル。
        :param optimizer: トレーニングに使用するオプティマイザ。
        :param scheduler: トレーニングに使用する学習率スケジューラ。
        """
        super().__init__()

        # この行により、'self.hparams'属性で初期化パラメータにアクセスできます
        # また、初期化パラメータがckptに保存されることを保証します
        self.save_hyperparameters(logger=False)

        self.net = net

        # 損失関数
        self.criterion = torch.nn.CrossEntropyLoss()

        # バッチ間で精度を計算し平均するためのメトリックオブジェクト
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # バッチ間で損失を平均するため
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # これまでの最高検証精度を追跡するため
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """モデル`self.net`を通して順伝播を実行します。

        :param x: 画像のテンソル。
        :return: ロジットのテンソル。
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """トレーニングが開始されるときに呼び出されるLightningフック。"""
        # デフォルトではlightningはトレーニング開始前に検証ステップの健全性チェックを実行するため、
        # 検証メトリクスがこれらのチェックからの結果を保存しないようにすることが重要です
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """データのバッチに対して単一のモデルステップを実行します。

        :param batch: データのバッチ（タプル）で、画像の入力テンソルとターゲットラベルを含みます。

        :return: 以下を含むタプル（順番に）：
            - 損失のテンソル。
            - 予測のテンソル。
            - ターゲットラベルのテンソル。
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """トレーニングセットからのデータのバッチに対して単一のトレーニングステップを実行します。

        :param batch: データのバッチ（タプル）で、画像の入力テンソルとターゲットラベルを含みます。
        :param batch_idx: 現在のバッチのインデックス。
        :return: モデルの予測とターゲット間の損失のテンソル。
        """
        loss, preds, targets = self.model_step(batch)

        # メトリクスを更新してログに記録
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # 損失を返さないとバックプロパゲーションが失敗します
        return loss

    def on_train_epoch_end(self) -> None:
        "トレーニングエポックが終了するときに呼び出されるLightningフック。"
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """検証セットからのデータのバッチに対して単一の検証ステップを実行します。

        :param batch: データのバッチ（タプル）で、画像の入力テンソルとターゲットラベルを含みます。
        :param batch_idx: 現在のバッチのインデックス。
        """
        loss, preds, targets = self.model_step(batch)

        # メトリクスを更新してログに記録
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "検証エポックが終了するときに呼び出されるLightningフック。"
        acc = self.val_acc.compute()  # 現在の検証精度を取得
        self.val_acc_best(acc)  # これまでの最高検証精度を更新
        # メトリックオブジェクトとしてではなく、`.compute()`メソッドを通じて値として`val_acc_best`をログに記録します
        # そうしないと、メトリックはエポックごとにlightningによってリセットされます
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """テストセットからのデータのバッチに対して単一のテストステップを実行します。

        :param batch: データのバッチ（タプル）で、画像の入力テンソルとターゲットラベルを含みます。
        :param batch_idx: 現在のバッチのインデックス。
        """
        loss, preds, targets = self.model_step(batch)

        # メトリクスを更新してログに記録
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """テストエポックが終了するときに呼び出されるLightningフック。"""
        pass

    def setup(self, stage: str) -> None:
        """fit（トレーニング＋検証）、validate、test、またはpredictの開始時に呼び出されるLightningフック。

        モデルを動的に構築したり、モデルについて何かを調整したりする必要がある場合に良いフックです。
        このフックはDDPを使用するときに全てのプロセスで呼び出されます。

        :param stage: `"fit"`、`"validate"`、`"test"`、または`"predict"`のいずれか。
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """最適化に使用するオプティマイザと学習率スケジューラを選択します。
        通常は1つ必要です。しかしGANや類似のものの場合、複数持つかもしれません。

        例：
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: トレーニングに使用するように設定されたオプティマイザと学習率スケジューラを含む辞書。
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)