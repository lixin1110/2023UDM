# encoding=utf8
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
#import xgboost as xgb
import random
import json, os
import torch
import torch.optim as optim
from UDM_model import UDM_model
from UDM2022.UDM.UDM_param.utils.getdata import My_dataset
from torch.utils.data import DataLoader
import timeit
from UDM2022.UDM.UDM_param.utils.tools import EarlyStopping, adjust_learning_rate
## 导入amp工具包


# device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_list = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='[UDM] Universality Distinction Mechanism')
parser.add_argument('--training_iters', type=int, default=1, help='train epochs')
parser.add_argument('--n_inputs', type=int, default=1, help='dimension of input')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type2',help='adjust learning rate')
parser.add_argument('--timestep', type=int, default=7, help='input sequence length of Informer encoder')
parser.add_argument('--output_len', type=int, default=7, help='start token length of Informer decoder')
parser.add_argument('--input_len', type=int, default=56, help='input length of UDM_model')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of UDM_model')
parser.add_argument('--trainsize', type=float, default=0.7, help='percent of train data')
parser.add_argument('--norm', type=str, default='N', help='data normalize or false')
parser.add_argument('--dropout', type=str, default=0.2, help='data normalize or not')
parser.add_argument('--BatchNorm', type=str, default='Y', help='bn y or n')
parser.add_argument('--adjust_lr', type=str, default='Y', help='bn y or n')
parser.add_argument('--pin', type=float, default=6, help='weight of pinball loss function')

args = parser.parse_args()
print('args:', args)
setting = 'itr{}_bs{}_pt{}_lr{}_lradj{}_ts{}_op{}_ip{}_hs{}_ts{}_nm{}_{}dp_{}bn_{}adj_{}pin_test'.format(
    args.training_iters, args.batch_size, args.patience, args.learning_rate, args.lradj, 
    args.timestep, args.output_len, args.input_len, args.hidden_size, 
    args.trainsize, args.norm, args.dropout, args.BatchNorm, args.adjust_lr, args.pin)

# input path
data_dir = ''
in_path = ''

# result path
output_dir = './result/'
output_forec = setting + '.csv'

# model path
PATH =  './model_dict/'
checkpoints =  './checkpoints/'
loss_dir = './log_file/' + setting + '/'


# %%

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, train_dataset, optimizer):
    model.train()
    train_x = train_dataset[0].to(device).unsqueeze(2)
    train_y = train_dataset[1].to(device).unsqueeze(2)
    train_x_mark = train_dataset[2].float().to(device)
    train_y_mark = train_dataset[3].float().to(device)
    optimizer.zero_grad()

    oneloss, mse = model.forward(train_x, train_y, train_x_mark, train_y_mark, True)
    oneloss.backward(torch.ones_like(oneloss))
    optimizer.step()

    return oneloss, mse

def validate(model, val_dataset):
    model.eval()
    with torch.no_grad():      
        val_x = val_dataset[0].to(device).unsqueeze(2)
        val_y = val_dataset[1].to(device).unsqueeze(2)
        val_x_mark = val_dataset[2].float().to(device)
        val_y_mark = val_dataset[3].float().to(device)
        oneloss, mse = model.forward(val_x, val_y, val_x_mark, val_y_mark, True)
    return oneloss, mse


def predict(model, test_dataset):    
    model.eval() 
    with torch.no_grad():
        test_x = test_dataset[0].to(device).float().unsqueeze(2) # add channel dimension
        test_x_mark = test_dataset[2].float().to(device) # add channel dimension
        test_y_mark = test_dataset[3].float().to(device) # add channel dimension
        output = model.module.predict(test_x, test_x_mark, test_y_mark)
        pred = output
    return pred


def back_zscore_normalization(predData, rec_scaler):
    for i,i_data in enumerate(predData):
        for j in range(len(i_data)):
            i_data[j] = float(i_data[j]*rec_scaler['std'] + rec_scaler['mean'])
    return predData


def pipeline():
    seed_torch()
    csv_result = pd.DataFrame(columns = ['id', 'V1','V2', 'V3', 'V4', 'V5', 'V6', 'V7'])
    
    fileList = os.listdir(data_dir)
    
    for f in fileList:
        print('仓库：', f.split('.')[0])
        in_file = in_path + f.replace('.csv', '.json')
        ids = []
        train_X, train_y = [], []
        X_ = []
        test_X, test_y = [], []
        idlist, actual = [], []
        for line in open(in_file, 'r', encoding='utf-8'):
            raw_data = json.loads(line)
            if raw_data['watch'] == 0:
                ids = [f+'-'+str(list(raw_data['id'])[i]) for i in range(len(list(raw_data['id'])))]
                idlist = idlist + ids
                test_X = test_X + [[raw_data['input_end_date'], list(np.array(raw_data['X']).transpose(1,0))]]
                test_y = test_y + list(np.array(raw_data['y']).transpose(1,0))
                actual = actual + np.array(raw_data['y']).transpose(1,0).tolist()
            else:
                tmp_x = np.array(raw_data['X']).transpose(1,0)
                tmp_y = np.array(raw_data['y']).transpose(1,0)
                train_X = train_X + [[raw_data['input_end_date'], list(tmp_x)]]
                train_y = train_y + list(tmp_y)
                for p in range(tmp_x.shape[0]):
                    X_ = np.concatenate((X_, tmp_x[p], tmp_y[p]),axis=0)
                    
        mean = np.mean(X_)
        std = np.std(X_,ddof=1)
        scaler = {'mean':mean, 'std':std}
        testdataset = My_dataset(test_X, test_y, scaler)
        fulldataset = My_dataset(train_X, train_y, scaler)
        train_size = int(args.trainsize * len(fulldataset))
        test_size = len(fulldataset) - train_size
        traindataset, valdataset = torch.utils.data.random_split(fulldataset, [train_size, test_size])
        data_loader_train = DataLoader(traindataset, batch_size=args.batch_size*2, shuffle=True, drop_last=True)
        data_loader_val = DataLoader(valdataset, batch_size=args.batch_size*2, shuffle=True, drop_last=True)
        data_loader_test = DataLoader(testdataset, batch_size=args.batch_size*2, shuffle=False, drop_last=False)
        start = timeit.default_timer()
        model = UDM_model(args.timestep, args.n_inputs, args.hidden_size).to(device_list[0])
        model = torch.nn.DataParallel(model, device_ids=device_list, output_device=device_list[0])
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        ckpath = os.path.join(checkpoints, setting)
        if not os.path.exists(ckpath):
            os.makedirs(ckpath)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        min_loss = 200
        rec_loss = []
        print('------------------------Train------------------------')
        for i in range(0, args.training_iters):
            # 训练
            sum_loss_train, sum_loss_val, sum_mse_train, sum_mse_val = 0, 0, 0, 0
            for i_batch, batch_data in enumerate(data_loader_train):
                train_loss, train_mse = train(model, batch_data, optimizer)
                if len(train_loss.shape)==0:
                    sum_loss_train = sum_loss_train + train_loss
                    sum_mse_train = sum_mse_train + train_mse
                else:
                    sum_loss_train = sum_loss_train + train_loss[0] + train_loss[1]
                    sum_mse_train = sum_mse_train + train_mse[0] + train_mse[1]
                    
                if i_batch%500 == 0 or i_batch == (len(data_loader_train)-1):
                    print('------train iter ', i_batch, ': ',torch.mean(train_loss))
            mean_train_loss = torch.mean(sum_loss_train) / len(data_loader_train)
            mean_train_mse = torch.mean(sum_mse_train) / len(data_loader_train)

            for i_batch, batch_data in enumerate(data_loader_val):
                val_loss, val_mse = validate(model, batch_data)
                if len(val_loss.shape)==0:
                    sum_loss_val = sum_loss_val + val_loss
                    sum_mse_val = sum_mse_val + val_mse
                else:
                    sum_loss_val = sum_loss_val + val_loss[0] + val_loss[1]
                    sum_mse_val = sum_mse_val + val_mse[0] + val_mse[1]
            mean_val_loss = torch.mean(sum_loss_val)/len(data_loader_val)
            mean_val_mse = torch.mean(sum_mse_val)/len(data_loader_val)
            rec_loss.append([mean_train_loss.tolist(), mean_train_mse.tolist(),
                             mean_val_loss.tolist(), mean_val_mse.tolist()])
            print('Epoch ', i+1, ', train: ', mean_train_loss, ', val: ', mean_val_loss)

            early_stopping(mean_val_loss, model, ckpath)
            #如果收敛性测试, 屏蔽此段, 不提前终止训练
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, i+1, args)

            if mean_val_loss < min_loss:
                min_loss = mean_val_loss
                print('min_loss:', min_loss)
                torch.save(model.module.state_dict(), PATH + 'best_cpc_ResLSTM.pkl', _use_new_zipfile_serialization=False)
                torch.save(optimizer.state_dict(), PATH + 'best_optimizer.pt', _use_new_zipfile_serialization=False)
        lossdf = pd.DataFrame(rec_loss,columns=['train_Pin-DWT','train_MSE', 'val_Pin-DWT','val_MSE'])
        if not os.path.isdir(loss_dir):
            os.makedirs(loss_dir)
        lossdf.to_csv(loss_dir + f, index=False)
        print('Train Finished!!!')
        # 预测
        print('------------Predict------------')
        model = UDM_model(args.timestep, args.n_inputs, args.hidden_size).to(device_list[0])
        model = torch.nn.DataParallel(model, device_ids=device_list, output_device=device_list[0])
        model.module.load_state_dict(torch.load(PATH + 'best_cpc_ResLSTM.pkl'))
        optimizer.load_state_dict(torch.load(PATH + 'best_optimizer.pt'))
        pred_result = []
        for i_batch, batch_data in enumerate(data_loader_test):
            pred_data = predict(model, batch_data).view(-1, args.output_len)
            pred_data = pred_data.cpu().numpy().tolist()
            pred_result = pred_result + pred_data

        back_pred_data = back_zscore_normalization(pred_result, testdataset.recscaler)
        end = timeit.default_timer()
        mse_, mae_, r2_, evar_ = [], [], [], []
        for i in range(len(back_pred_data)):
            i_pred = back_pred_data[i]
            for j in range(len(i_pred)):
                if i_pred[j] < 0:
                    i_pred[j] = 0
            i_actual = actual[i]
            id = idlist[i]
            MSE = mean_squared_error(i_actual, i_pred, sample_weight=None, multioutput='uniform_average')
            MAE = mean_absolute_error(i_actual, i_pred, sample_weight=None, multioutput='uniform_average')
            R_squared = r2_score(i_actual, i_pred)
            EVAR = explained_variance_score(i_actual, i_pred, sample_weight=None, multioutput='uniform_average')
            mse_.append(MSE)
            mae_.append(MAE)
            r2_.append(R_squared)
            evar_.append(EVAR)
            series_actual = pd.Series({"id":id, "V1":i_actual[0], "V2":i_actual[1], "V3":i_actual[2], "V4":i_actual[3], "V5":i_actual[4], "V6":i_actual[5], "V7":i_actual[6], "MSE":MSE, "MAE":MAE, "R_squared":R_squared, "Explained_var":EVAR}, name="actual")
            series_pred = pd.Series({"id":id, "V1":i_pred[0], "V2":i_pred[1], "V3":i_pred[2], "V4":i_pred[3], "V5":i_pred[4], "V6":i_pred[5], "V7":i_pred[6]}, name="pred")
            csv_result = csv_result.append(series_actual)
            csv_result = csv_result.append(series_pred)
        print('仓库：', f.split('.')[0])
        print('Runtime: ', str(end-start))
        print('MSE:', round(np.mean(mse_), 4), '\t MAE:', round(np.mean(mae_), 4), '\t R_squared:', round(np.mean(r2_),4), '\t Estimated_var:', round(np.mean(evar_),4))

    # 生成结果文件
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + output_forec
    csv_result.to_csv(output_file, float_format='%.3f', index=True)
    print("完成！")


if __name__ == "__main__":
    pipeline()
    

# %%
