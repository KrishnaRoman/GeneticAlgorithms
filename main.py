
# different loss functions
def dice_coef(y_true, y_pred):
    smooth = 1.0  #0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

def dice_coef_loss(y_true,y_pred):
    return 1 - dice_coef(y_true,y_pred)

def iou_loss(y_true,y_pred):
    return 1 - jacard(y_true, y_pred)

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.75
    smooth = 1
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def saveModel(model):

    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass

    fp = open('models/modelP.json','w')
    fp.write(model_json)
    model.save('models/modelW.h5')

def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):
    dice_fold = 0
    jacard_fold = 0
    best = -1

    def evaluateModel(model, X_test, Y_test, batchSize):
        nonlocal dice_fold, jacard_fold, best

#        try:
#            os.makedirs('results')
#        except:
#            pass

        yp = model.predict(x=X_test, batch_size=batchSize, verbose=0)
        yp = np.round(yp,0)

        jacard = 0
        dice = 0

        for i in range(len(Y_test)):
            yp_2 = yp[i].ravel()
            y2 = Y_test[i].ravel()

            intersection = yp_2 * y2
            union = yp_2 + y2 - intersection

            jacard += (np.sum(intersection)/np.sum(union))

            dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))

        jacard /= len(Y_test)
        dice /= len(Y_test)

        #print('Jacard Index : '+str(jacard))
        #print('Dice Coefficient : '+str(dice))

        if(jacard>float(best)):
            #print('***********************************************')
            #print('Jacard Index improved from '+str(best)+' to '+str(jacard))
            #print('***********************************************')

            best = jacard

#            saveModel(model)

            dice_fold = dice
            jacard_fold = jacard

    for epoch in range(epochs):
        #print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=0)

        evaluateModel(model,X_test, Y_test,batchSize)

    return model, dice_fold, jacard_fold


def binary_tournament(pop, P, _, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    # the result this function returns
    import numpy as np
    S = np.full(n_tournaments, -1, dtype=np.int)

    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]

        # if the first individual is better, choose it
        if pop[a].F < pop[b].F:
            S[i] = a

        # otherwise take the other individual
        else:
            S[i] = b

    return S

if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import dill
    from sklearn import preprocessing
    from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
    from pymoo.core.variable import Real, Integer, Choice, Binary
    from pymoo.core.mixed import MixedVariableGA
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    from pymoo.termination.max_gen import MaximumGenerationTermination
    from pymoo.optimize import minimize
    from pymoo.operators.selection.tournament import TournamentSelection
    from keras.models import Model, model_from_json
    from tensorflow.keras.optimizers import Adam, Nadam

    from keras import backend as K
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from keras import applications, optimizers, callbacks
    import keras
    from model import DCUNet
    from sklearn.model_selection import KFold
    import multiprocessing

    from pymoo.core.problem import DaskParallelization
    import dask
    from dask.distributed import Client, LocalCluster
    from dask_cuda import LocalCUDACluster
    import sys
    import gc
    import os
    import cv2

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    X = []
    Y = []

    for i in range(612):
        path = 'CVC-ClinicDB/Original/'+ str(i+1)+'.tif'
        img = cv2.imread(path,1)
        resized_img = cv2.resize(img,(128, 96), interpolation = cv2.INTER_CUBIC)

        X.append(resized_img)

    for i in range(612):
        path2 = 'CVC-ClinicDB/Ground Truth/' + str(i+1)+'.tif'
        msk = cv2.imread(path2,0)

        resized_msk = cv2.resize(msk,(128, 96), interpolation = cv2.INTER_CUBIC)

        Y.append(resized_msk)

    X = np.array(X)
    Y = np.array(Y)

    Y = Y.reshape((Y.shape[0],Y.shape[1],Y.shape[2],1))

    X = X/255
    Y = Y/255

    Y=np.round(Y,0)
    print(Y.shape)

    class MixedVariableProblem(ElementwiseProblem):

        def __init__(self, **kwargs):
            vars = {
                "batch_size": Integer(bounds=(1, 3)),
                "dropout": Real(bounds=(0, 1)),
                "learning_rate": Real(bounds=(-4, -2)),
                "beta_1": Real(bounds=(0.8, 1)),
                "beta_2": Real(bounds=(0.8, 1))
            }
            super().__init__(vars=vars, n_obj=1, **kwargs)

        def _evaluate(self, Xh, out, *args, **kwargs):
            # Define per-fold score containers
            dice_per_fold = [0]*5
            jacard_per_fold = [0]*5
            kfold = KFold(n_splits=5, shuffle=True)

            fold_no = 0
            for train, test in kfold.split(X, Y):

              batch_size = 2**Xh["batch_size"]
              learning_rate = 10**Xh["learning_rate"]
              optimizer = Nadam(learning_rate=Xh["learning_rate"], beta_1=Xh["beta_1"], beta_2=Xh["beta_2"])

              model = DCUNet(height=96, width=128, channels=3, dropout=Xh["dropout"])
              model.compile(optimizer=optimizer, loss=focal_tversky, metrics=[dice_coef, jacard, 'accuracy'])
              _, dice_per_fold[fold_no], jacard_per_fold[fold_no] = trainStep(model, X[train], Y[train], X[test], Y[test], epochs=100, batchSize=batch_size)
              fold_no = fold_no + 1

              print("evaluation done! FOLD", fold_no)

              tf.keras.backend.clear_session()
              tf.compat.v1.reset_default_graph()
              gc.collect()
            out["F"] = 1 - np.mean(jacard_per_fold)

    n_workers=5
    threads_per_worker=1

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1,2,3,4",
				n_workers=n_workers,
				threads_per_worker=threads_per_worker)

    with dask.config.set({"distributed.scheduler.worker-saturation": 1.0}):
        client = Client(cluster)
        client.restart()
        print("DASK STARTED")

        runner = DaskParallelization(client)

        problem = MixedVariableProblem(elementwise_runner=runner)

        algorithm = MixedVariableGA(pop_size=50, selection=TournamentSelection(func_comp=binary_tournament))

        termination = get_termination("n_gen", 10)

        res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   copy_algorithm=False,
                   save_history=True,
                   verbose=True)

        print(f"Best solution found: \nX = {res.X}\nF = {1-res.F}\nCV= {res.CV}")
        print(f"{n_workers} GPUs - {threads_per_worker} threads_per_worker: {res.exec_time}")

        print("DASK SHUTDOWN")

        #with open("checkpoint", "wb") as f:
        #    dill.dump(algorithm, f)

        client.close()
