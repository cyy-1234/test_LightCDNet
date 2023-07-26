import numpy as np
import  cv2
import os
def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    # mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    # print(iu)
    # mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    print("precision:{},recall:{},F1:{},iou:{},OA:{}".format(precision[1],recall[1],F1[1],iu[1],acc))
    # score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}

    # return score_dict

def get_confuse_matrix(num_classes, label_gts_path, label_preds_path,names):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for name in names:
        lt=cv2.imread(((os.path.join(label_gts_path,name))),0)
        lt = lt.clip(max=1)
        lp=cv2.imread(((os.path.join(label_preds_path,name))),0)
        lp = lp.clip(max=1)
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix

if __name__ == '__main__':

    # Note! !!
    # Make sure that the bit depth of the real label and the prediction graph are consistent, both are uint8.
    label_gts_path= r'LEVIR-CD/test/label'
    # label_gts_path=r'Z:\cyy\bigpaper\panyu-cd\select\bigpaper\2013-2019\test\label'
    # label_gts_path=r'F:\lab\WHU\256\test\label'
    # label_gts_path=r'F:\lab\DSIFN-Dataset\256\test\label'
    # label_gts_path=r'F:\lab\255pair_png\test\label_256'
    # label_preds_path=r'F:\lab\pytorch-deeplab-xception\LEVIR-CD_mobilenet_deconv_batchsize16_epoch200'
    # label_preds_path=r'F:\lab\pytorch-deeplab-xception\WHU_mobilenet_deconv_convx1_batchsize8_epoch100_newest'
    # label_preds_path=r'F:\lab\pytorch-deeplab-xception\DSIFN_mobilenet_deconv_batchsize16_epoch200'
    # label_preds_path=r'F:\lab\pytorch-deeplab-xception\LEVIR-CD_mobilenet_deconv_convx1_batchsize8_epoch100_newest'
    label_preds_path=r'LEVIR'
    names=os.listdir(label_preds_path)
    print(len(names))
    num_classes=2
    confusion_matrix=get_confuse_matrix(num_classes, label_gts_path, label_preds_path,names)
    cm2score(confusion_matrix)
# 224
# precision:0.9142208003465182,recall:0.8717010140105654,F1:0.8924546854074382,iou:0.8057952855035397,OA:0.9892975315451613
# 256
# precision:0.9132862070650856,recall:0.8908533121634898,F1:0.9019302333178941,iou:0.8213779936669691,OA:0.9901308566331855

# 224
# precision:0.914,recall:0.872,F1:0.892,iou:0.806,OA:0.989
# 256
# precision:0.913,recall:0.891,F1:0.902,iou:0.821,OA:0.990
