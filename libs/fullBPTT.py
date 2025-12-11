#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn
from eipl.utils import LossScheduler


class fullBPTTtrainerTactile:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """
    #クラスの初期化関数
    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0], device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")#学習中に損失の重みを徐々に変化させる関数
        self.model = model.to(self.device)
    #学習したモデルをファイルに保存する関数
    def save(self, epoch, loss, savename):
        torch.save(
            {
                "epoch": epoch, #何回目の学習か
                "model_state_dict": self.model.state_dict(), #モデルのパラメータ(重みとバイアス)
                #'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],#学習データの損失
                "test_loss": loss[1],#テストデータの損失
            },
            savename,#savenameはディレクトリパス。SARNN.pthというファイル名で保存
        )
    #1回分（エポック）の学習を実行する関数
    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()#推論モードにする(ドロップアウトなどを無効化)

        total_loss = 0.0
        total_img_loss = 0.0
        total_img_tactile_loss = 0.0
        total_joint_loss = 0.0
        total_pt_loss = 0.0
        total_pt_tactile_loss = 0.0
        for n_batch, ((x_img, x_joint, x_img_tactile), (y_img, y_joint, y_img_tactile)) in enumerate(data):#n_batchはバッチ番号、((x_img, x_joint), (y_img, y_joint))はバッチデータ。enumerateはインデックス番号と値を同時に取得可能。xが入力データで、yが正解データ
            if "cpu" in self.device:
                x_img = x_img.to(self.device)#データをcpuに移動
                y_img = y_img.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)
                x_img_tactile = x_img_tactile.to(self.device)
                y_img_tactile = y_img_tactile.to(self.device)

            state = None#RNNの初期状態
            yi_list, yv_list, yt_list = [], [], []#予測結果のリスト
            dec_pts_list, enc_pts_list = [], []
            dec_tactile_pts_list, enc_tactile_pts_list = [], []
            self.optimizer.zero_grad(set_to_none=True)#前回の勾配をリセット
            for t in range(x_img.shape[1] - 1):#x_img.shape[1]はバッチサイズ、これで時系列の長さという意味。-1は最初のデータは入力として使わないため
                _yi_hat, _yv_hat, enc_ij, dec_ij, state, _yt_hat, enc_tactile_ij,dec_tactile_ij = self.model(
                    x_img[:, t], x_joint[:, t], x_img_tactile[:,t], state
                )#モデルの予測を行う。_yi_hatは画像の予測、_yv_hatは関節の予測、enc_ijは画像の潜在表現、dec_ijは関節の潜在表現、stateはRNNの状態
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                yt_list.append(_yt_hat)
                enc_pts_list.append(enc_ij)
                dec_pts_list.append(dec_ij)
                enc_tactile_pts_list.append(enc_tactile_ij)
                dec_tactile_pts_list.append(dec_tactile_ij)                
            #torch.stackはリストをテンソルに変換。torch.permute(リスト,タプル)でテンソルの軸を入れ替え。(1, 0, 2, 3, 4)はテンソルの形状を指定。1はバッチサイズ、0は時系列の長さ、2はチャンネル数、3は高さ、4は幅
            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))#予測画像をpytorchで使える形の（バッチ数、時刻、チャンネル、高さ、幅）のテンソルに変換
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))#予測関節をpytorchで使える形の(バッチ数、時刻、関節(9個))の形に変換
            yt_hat = torch.permute(torch.stack(yt_list), (1, 0, 2, 3, 4))

            img_loss = nn.MSELoss()(yi_hat, y_img[:, 1:]) * self.loss_weights[0]#画像の損失を計算(yi_hatは予測画像、y_img[:, 1:]は正解画像)
            img_tactile_loss = nn.MSELoss()(yt_hat,y_img_tactile[:,1:]) * self.loss_weights[3]
            joint_loss = nn.MSELoss()(yv_hat, y_joint[:, 1:]) * self.loss_weights[1]#関節の損失を計算(yv_hatは予測関節、y_joint[:, 1:]は正解関節)
            # Gradually change the loss value using the LossScheluder class.pt_lossは注意点のロス
            pt_loss = nn.MSELoss()(
                torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])
            ) * self.scheduler(self.loss_weights[2])

            pt_tactile_loss = nn.MSELoss()(
                torch.stack(dec_tactile_pts_list[:-1]), torch.stack(enc_tactile_pts_list[1:])
            ) * self.scheduler(self.loss_weights[4])
            loss = img_loss + joint_loss + pt_loss + img_tactile_loss + pt_tactile_loss
            
            total_loss += loss.item()
            total_img_loss += img_loss.item()
            total_joint_loss += joint_loss.item()
            total_pt_loss += pt_loss.item()
            total_img_tactile_loss += img_tactile_loss.item()
            total_pt_tactile_loss += pt_tactile_loss.item()

            if training:
                loss.backward()#勾配計算
                self.optimizer.step()#勾配を用いてパラメータを更新

        n_batches = n_batch + 1
        return {
            'total_loss': total_loss / n_batches,
            'img_loss': total_img_loss / n_batches,
            'joint_loss': total_joint_loss / n_batches,
            'pt_loss': total_pt_loss / n_batches,
            'tactile_loss': total_img_tactile_loss / n_batches,
            'pt_tactile_loss': total_pt_tactile_loss / n_batches
        }
    

class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """
    #クラスの初期化関数
    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0], device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")#学習中に損失の重みを徐々に変化させる関数
        self.model = model.to(self.device)
    #学習したモデルをファイルに保存する関数
    def save(self, epoch, loss, savename):
        torch.save(
            {
                "epoch": epoch, #何回目の学習か
                "model_state_dict": self.model.state_dict(), #モデルのパラメータ(重みとバイアス)
                #'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],#学習データの損失
                "test_loss": loss[1],#テストデータの損失
            },
            savename,#savenameはディレクトリパス。SARNN.pthというファイル名で保存
        )
    #1回分（エポック）の学習を実行する関数
    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()#推論モードにする(ドロップアウトなどを無効化)

        total_loss = 0.0
        total_img_loss = 0.0
        # total_img_tactile_loss = 0.0
        total_joint_loss = 0.0
        total_pt_loss = 0.0
        # total_pt_tactile_loss = 0.0
        for n_batch, ((x_img, x_joint), (y_img, y_joint)) in enumerate(data):#n_batchはバッチ番号、((x_img, x_joint), (y_img, y_joint))はバッチデータ。enumerateはインデックス番号と値を同時に取得可能。xが入力データで、yが正解データ
            if "cpu" in self.device:
                x_img = x_img.to(self.device)#データをcpuに移動
                y_img = y_img.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)

            state = None#RNNの初期状態
            yi_list, yv_list, yt_list = [], [], []#予測結果のリスト
            dec_pts_list, enc_pts_list = [], []
            self.optimizer.zero_grad(set_to_none=True)#前回の勾配をリセット
            for t in range(x_img.shape[1] - 1):#x_img.shape[1]はバッチサイズ、これで時系列の長さという意味。-1は最初のデータは入力として使わないため
                _yi_hat, _yv_hat, enc_ij, dec_ij, state= self.model(
                    x_img[:, t], x_joint[:, t], state
                )#モデルの予測を行う。_yi_hatは画像の予測、_yv_hatは関節の予測、enc_ijは画像の潜在表現、dec_ijは関節の潜在表現、stateはRNNの状態
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                enc_pts_list.append(enc_ij)
                dec_pts_list.append(dec_ij)               
            #torch.stackはリストをテンソルに変換。torch.permute(リスト,タプル)でテンソルの軸を入れ替え。(1, 0, 2, 3, 4)はテンソルの形状を指定。1はバッチサイズ、0は時系列の長さ、2はチャンネル数、3は高さ、4は幅
            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))#予測画像をpytorchで使える形の（バッチ数、時刻、チャンネル、高さ、幅）のテンソルに変換
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))#予測関節をpytorchで使える形の(バッチ数、時刻、関節(9個))の形に変換

            img_loss = nn.MSELoss()(yi_hat, y_img[:, 1:]) * self.loss_weights[0]#画像の損失を計算(yi_hatは予測画像、y_img[:, 1:]は正解画像)
            joint_loss = nn.MSELoss()(yv_hat, y_joint[:, 1:]) * self.loss_weights[1]#関節の損失を計算(yv_hatは予測関節、y_joint[:, 1:]は正解関節)
            # Gradually change the loss value using the LossScheluder class.pt_lossは注意点のロス
            pt_loss = nn.MSELoss()(
                torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])
            ) * self.scheduler(self.loss_weights[2])

            loss = img_loss + joint_loss + pt_loss
            
            total_loss += loss.item()
            total_img_loss += img_loss.item()
            total_joint_loss += joint_loss.item()
            total_pt_loss += pt_loss.item()

            if training:
                loss.backward()#勾配計算
                self.optimizer.step()#勾配を用いてパラメータを更新

        n_batches = n_batch + 1
        return {
            'total_loss': total_loss / n_batches,
            'img_loss': total_img_loss / n_batches,
            'joint_loss': total_joint_loss / n_batches,
            'pt_loss': total_pt_loss / n_batches
        }
