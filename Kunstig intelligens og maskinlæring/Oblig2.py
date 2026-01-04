import numpy as np
import matplotlib.pyplot as plt
import sklearn # This is only to generate a dataset
import pandas as pd
from sklearn.metrics import precision_score, recall_score


# Generating the dataset
from sklearn.datasets import make_blobs
X, t_multi = make_blobs(n_samples=[400, 400, 400, 400, 400], centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
                  n_features=2, random_state=2024, cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5])

# Shuffling the dataset
indices = np.arange(X.shape[0])
rng = np.random.RandomState(2024)
rng.shuffle(indices)
indices[:10]


# Splitting into train, dev and test
X_train = X[indices[:1000],:]
X_val = X[indices[1000:1500],:]
X_test = X[indices[1500:],:]
t_multi_train = t_multi[indices[:1000]]
t_multi_val = t_multi[indices[1000:1500]]
t_multi_test = t_multi[indices[1500:]]

t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int')

plt.figure(figsize=(8,6)) # You may adjust the size
plt.scatter(X_train[:, 0], X_train[:, 1], c=t_multi_train, s=10.0)
plt.title("Multi-class set")

plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=10.0)
plt.title("Binary set")


def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a prediction of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

# Task 1

def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

class NumpyClassifier():
    """Common methods to all Numpy classifiers --- if any"""
    

class NumpyLinRegClass(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias=bias
    
    def fit(self, X_train, t_train, lr = 0.1, epochs=10):
        """X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N,
        the target class values for the training data
        lr is our learning rate
        """
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, M) = X_train.shape
        
        self.weights = weights = np.zeros(M)
        
        for epoch in range(epochs):
            # print("Epoch", epoch)
            weights -= lr / N *  X_train.T @ (X_train @ weights - t_train)      
    
    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold
    
def accuracy(predicted, gold):
    return np.mean(predicted == gold)


class StandardScaler:
    """
    Skalerer dataen slik at kollonnene/features får gjennomsnitt 0 og standaradavvik 1
    """
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)+ 1e-8#unngå deling på 0

    
    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X): #setter de to forrige meotdene sammen
        self.fit(X)
        return self.transform(X)



#Logistic regression

def sigmoid(z):#får alle verdier til å bli mellom (0,1)
    z = np.clip(z, -50, 50) #begrenser z verdier til -50 og 50 ettersom jeg innimellom fikk veldig høye/lave verdier som førte til overflow/underflow
    return 1.0 / (1.0 + np.exp(-z)) #sigmoid : 1 / (1 + e^(-z))

class NumpyLogRegClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias = bias
        self.weights = None
        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.n_epochs_trained = 0  

    def fit(self, X_train, t_train, lr=0.1, epochs=10, X_val=None, t_val=None,
            tol=1e-4, n_epochs_no_update=10):
        if self.bias: #legger til bias hvis det finnes
            X_train = add_bias(X_train, self.bias)
        (N, M) = X_train.shape
        self.weights = weights = np.zeros(M)
        self.loss_history = [] #for hvis X_val ikke er gitt
        self.acc_history = []#for hvis X_val ikke er gitt
        self.val_loss_history = []#for hvis X_val er gitt
        self.val_acc_history = []#for hvis X_val er gitt
        self.n_epochs_trained = 0

        if X_val is not None and t_val is not None and self.bias:
            X_val_b = add_bias(X_val, self.bias)#legger til bias hos val hvis det er gitt
        else: 
            X_val_b = X_val

        best_loss = np.inf 
        no_improve = 0

        for _ in range(epochs):
            yhat = sigmoid(X_train @ weights)  # sannsynligheter mellom (0,1)

            # BCE binary cross-entropy
            loss = -np.mean(t_train * np.log(yhat+ 1e-12) + (1 - t_train) * np.log(1 - yhat+ 1e-12))
            self.loss_history.append(loss)

            # accuracy på trening
            self.acc_history.append(np.mean((yhat > 0.5) == t_train))

            # gradient (BCE)
            grad = (X_train.T @ (yhat - t_train)) / N #gradient
            weights -= lr * grad #lage nye vekter ved bruk av læringsraten og gradienten

            # validering
            if X_val_b is not None:  #hvis X_val er gitt legger man til losses og accuracy i 
                yv = sigmoid(X_val_b @ weights)
                vloss = -np.mean(t_val * np.log(yv) + (1 - t_val) * np.log(1 - yv))
                self.val_loss_history.append(vloss)
                self.val_acc_history.append(np.mean((yv > 0.5) == t_val))
                monitor_loss = vloss
            else:
                monitor_loss = loss

            # tidlig stopping 
            if monitor_loss + tol < best_loss:
                best_loss = monitor_loss
                no_improve = 0
            else:
                no_improve += 1

            self.n_epochs_trained += 1
            if no_improve >= n_epochs_no_update:
                break

    def predict_probability(self, X):
        if self.bias:
            X = add_bias(X, self.bias)
        return sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        if self.bias:
            X = add_bias(X, self.bias)
        probs = sigmoid(X @ self.weights)
        return (probs > threshold).astype(int)


    
class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std  = np.std(X, axis=0) + 1e-8
    def transform(self, X):
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    


def find_the_best(
    
    X_train, t_train, X_val, t_val,
    lrs, max_epochs_list,
    use_scalings=(True,),
    mode="binary",
    hidden_list=(6,),        
    tols=(1e-4,),            
    early_stops=(10,)         
):
    """
    Hoveddel av tuning oppgaven. Tester for kombinasjoner av gitte verdier for parametrene og lager en dictionary av den beste modellen.
    Velger best uifra accuracy.
    Støtter:
      - mode="binary"  -> NumpyLogRegClass
      - mode="softmax" -> SoftmaxClass
      - mode="ovr"     -> NumpyMultiLogRegClass
      - mode="mlp"     -> MLPBinaryLinRegClass  
      - mode="ovr_mlp" -> MLPOvR
    """
    best = None

    def log_result(**kw): #jeg har skrudd av logging (utenom mlp) ettersom jeg bare brukte det til testing underveis
        # felles, ryddig print
        msg = " - ".join(" " + k + " " + str(v) for k, v in kw.items())
        print(msg)

    for use_scaling in use_scalings: #sjekker om man skal bruke scaling
        if use_scaling:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train)
            Xva = scaler.transform(X_val)
        else:
            scaler = None
            Xtr, Xva = X_train, X_val

        
        if mode in ("binary", "softmax", "ovr", "mlp", "ovr_mlp"):
            for lr in lrs:
                for tol in tols:
                    for early_stop in early_stops:
                        for max_ep in max_epochs_list:

                            if mode == "binary":
                                clf = NumpyLogRegClass(bias=-1)
                                clf.fit(Xtr, t_train, lr=lr, epochs=max_ep,
                                        X_val=Xva, t_val=t_val,
                                        tol=tol, n_epochs_no_update=early_stop)

                                val_loss = (clf.val_loss_history[-1]
                                            if clf.val_loss_history
                                            else clf.loss_history[-1])
                                val_acc = (clf.val_acc_history[-1]
                                           if clf.val_acc_history else None) or 0.0

                                # log_result(mode=mode, scaled=use_scaling, lr=lr, tol=tol,
                                #            early_stop=early_stop, max_ep=max_ep,
                                #            trained_ep=clf.n_epochs_trained,
                                #            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

                                if (best is None) or (val_loss < best["val_loss"]):
                                    best = {
                                        "mode": mode, "scaled": use_scaling, "scaler": scaler,
                                        "lr": lr, "tol": tol, "early_stop": early_stop,
                                        "max_ep": max_ep, "trained_ep": clf.n_epochs_trained,
                                        "val_loss": val_loss, "val_acc": val_acc, "model": clf,
                                    }

                            elif mode == "softmax":
                                sm = SoftmaxClass(bias=-1)
                                sm.fit(Xtr, t_train, Xva, t_val,
                                       lr=lr, epochs=max_ep, tol=tol,
                                       n_epochs_no_update=early_stop)

                                val_acc = np.mean(sm.predict(Xva) == t_val)
                                classes = sm.classes_
                                idx = np.searchsorted(classes, t_val)
                                Tva = np.eye(len(classes))[idx]
                                Pv = sm.predict_proba(Xva)
                                val_loss = -np.mean(np.sum(Tva * np.log(Pv + 1e-12), axis=1))

                                # log_result(mode=mode, scaled=use_scaling, lr=lr, tol=tol,
                                #            early_stop=early_stop, max_ep=max_ep,
                                #            trained_ep=sm.n_epochs_trained,
                                #            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

                                if (best is None) or (val_acc > best["val_acc"]) or \
                                   (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
                                    best = {
                                        "mode": mode, "scaled": use_scaling, "scaler": scaler,
                                        "lr": lr, "tol": tol, "early_stop": early_stop,
                                        "max_ep": max_ep, "trained_ep": sm.n_epochs_trained,
                                        "val_loss": val_loss, "val_acc": val_acc,
                                        "model": sm, "classes": classes,
                                    }

                            elif mode == "ovr":
                                m = NumpyMultiLogRegClass(bias=1)
                                m.make_clfs(Xtr, t_train, lr=lr, epochs=max_ep,
                                            X_val=Xva, t_multi_val=t_val,
                                            tol=tol, n_epochs_no_update=early_stop)

                                val_acc = m.prediction_accuracy(Xva, t_val)
                                vlosses = [c.val_loss_history[-1] if c.val_loss_history else c.loss_history[-1]
                                           for c in m.clfs]
                                val_loss = float(np.mean(vlosses))
                                trained_ep_mean = float(np.mean([c.n_epochs_trained for c in m.clfs]))

                                # log_result(mode=mode, scaled=use_scaling, lr=lr, tol=tol,
                                #            early_stop=early_stop, max_ep=max_ep,
                                #            trained_ep=f"{trained_ep_mean:.1f}",
                                #            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

                                if (best is None) or (val_acc > best["val_acc"]) or \
                                   (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
                                    best = {
                                        "mode": mode, "scaled": use_scaling, "scaler": scaler,
                                        "lr": lr, "tol": tol, "early_stop": early_stop,
                                        "max_ep": max_ep, "trained_ep_mean": trained_ep_mean,
                                        "val_loss": val_loss, "val_acc": val_acc,
                                        "model": m, "classes": m.classes_,
                                    }

                            
                            elif mode == "ovr_mlp":
                                for dh in hidden_list:
                                    m = MLPOvR(bias=-1, dim_hidden=dh)
                                    m.fit(Xtr, t_train,
                                        lr=lr, epochs=max_ep,
                                        X_val=Xva, t_val=t_val,
                                        tol=tol, n_epochs_no_update=early_stop)

                                    val_acc = float(np.mean(m.predict(Xva) == t_val)) if Xva is not None else 0.0
                                    vlosses = [
                                        (c.val_loss_history[-1] if c.val_loss_history else c.loss_history[-1])
                                        for c in m.clfs
                                    ]
                                    val_loss = float(np.mean(vlosses)) if len(vlosses) > 0 else float("inf")
                                    trained_ep = float(np.mean([c.n_epochs_trained for c in m.clfs]))

                                    # log_result(mode=mode, scaled=use_scaling, lr=lr, tol=tol,
                                    #         early_stop=early_stop, max_ep=max_ep,
                                    #         trained_ep=f"{trained_ep:.1f}",
                                    #         dim_hidden=dh,
                                    #         val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

                                    # velg best: høyest acc, tie → lavest loss
                                    if (best is None) or (val_acc > best["val_acc"]) or \
                                    (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
                                        best = {
                                            "mode": mode, "scaled": use_scaling, "scaler": scaler,
                                            "lr": lr, "tol": tol, "early_stop": early_stop,
                                            "max_ep": max_ep, "trained_ep": trained_ep,
                                            "dim_hidden": dh, "val_loss": val_loss, "val_acc": val_acc,
                                            "model": m, "classes": m.classes_,
                                        }

                            elif mode == "mlp":
                                for dh in hidden_list:
                                    clf = MLPBinaryLinRegClass(bias=-1, dim_hidden=dh)
                                    clf.fit(Xtr, t_train,
                                            lr=lr, epochs=max_ep,
                                            X_val=Xva, t_val=t_val,
                                            tol=tol, n_epochs_no_update=early_stop)

                                    if clf.val_loss_history:
                                        val_loss = float(clf.val_loss_history[-1])
                                        val_acc  = float(clf.val_acc_history[-1]) if clf.val_acc_history else accuracy(clf.predict(Xva), t_val)
                                    else:
                                        val_loss = float(clf.loss_history[-1]) if clf.loss_history else float("inf")
                                        val_acc  = accuracy(clf.predict(Xva), t_val) if Xva is not None else 0.0

                                    trained_ep = float(clf.n_epochs_trained)

                                    log_result(mode=mode, scaled=use_scaling, lr=lr, tol=tol,
                                                early_stop=early_stop, max_ep=max_ep,
                                                trained_ep=f"{trained_ep:.1f}",
                                                dim_hidden=dh,
                                                val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

                                    if (best is None) or (val_acc > best["val_acc"]) or \
                                       (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
                                        best = {
                                            "mode": mode, "scaled": use_scaling, "scaler": scaler,
                                            "lr": lr, "tol": tol, "early_stop": early_stop,
                                            "max_ep": max_ep, "trained_ep": trained_ep,
                                            "dim_hidden": dh, "val_loss": val_loss, "val_acc": val_acc,
                                            "model": clf,
                                        }

        else:
            raise ValueError("mode må være 'binary', 'ovr', 'softmax' eller 'mlp'") #hvis man ikke velger en av classiferene 

    return best

class NumpyMultiLogRegClass(NumpyClassifier):
    """
    Minimal OvR-wrapper:
      - make_clfs: trener én logreg per klasse (k vs rest)
      - find_probs: returnerer (N, K) sannsynligheter
      - predictions: argmax over kolonner -> klasser
      - prediction_accuracy: treffprosent gitt fasit
    """
    def __init__(self, bias=1):
        self.bias = bias
        self.clfs = []
        self.classes_ = None

    def make_clfs(self, X_train, t_multi_train, lr=0.1, epochs=100,
                  X_val=None, t_multi_val=None, tol=1e-4, n_epochs_no_update=10):
        self.classes_ = np.unique(t_multi_train) #finner alle klassen og lagrer som liste med forskjellige tall(klasser) [0,1,2,3,4] feks
        self.clfs = []
        for k in self.classes_:
            t_k_tr = (t_multi_train == k).astype(int) #gjør om fasiten til 1-ere og 0-ere istedenfor mange froskjellige klasser for den klassen det gjelder
            t_k_va = None if (X_val is None or t_multi_val is None) else (t_multi_val == k).astype(int) #samme hvis man  har gitt t_multi_val
            clf = NumpyLogRegClass(bias=self.bias)
            clf.fit(X_train, t_k_tr,
                    lr=lr, epochs=epochs,
                    X_val=X_val, t_val=t_k_va,
                    tol=tol, n_epochs_no_update=n_epochs_no_update)
            self.clfs.append(clf)
        return self

    def find_probs(self, X):
        P_cols = [clf.predict_probability(X) for clf in self.clfs]#lager sannsynlighets arrayer for hvor sannsynlig det er for at punktet tilhører klassen for alle clfs
        return np.column_stack(P_cols)#lager det til matrise

    def predictions(self, X):
        P = self.find_probs(X)
        idx = np.argmax(P, axis=1) # gir alle punkter en predicted klasse
        return self.classes_[idx]

    def prediction_accuracy(self, X, t_multi):
        y_pred = self.predictions(X) #sjekker hvor bra predctions var forhold til fasit
        return np.mean(y_pred == t_multi)

    def predict(self, X): #trengs for plot funksjonen
        return self.predictions(X)


#softmax
def _softmax(Z):# softmax radvis
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True) #sannsynlighetsfordeling per datapunkt - feks fra [1.2, 0.3, -2.1] -> [0.63, 0.26, 0.11]

class SoftmaxClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias = bias
        self.weights = None
        self.classes_ = None
        self.n_epochs_trained = 0
        self.loss_history = []


    def _new_answers(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y) # legger til klassene
        idx = np.searchsorted(self.classes_, y)# passer på at alle klassene har riktig indeks i forhold til classes_ i tilfelle de ikke begynner på 0. hvis man har [0,1,3]->[0,1,2]
        K = len(self.classes_)
        Y = np.eye(K)[idx] #lager matrise med hvert punkt som rader og et 1 tall for hvilke klasse. hvis man har punkter som tilhører klassene (2,0,1) -> [[0,0,1][1,0,0][0,1,0]]
        return Y

    def fit(self, X_train, t_train, X_val, t_val,
            lr=0.1, epochs=100, tol=1e-4, n_epochs_no_update=10):

        Xtr = add_bias(X_train, self.bias) if self.bias else X_train
        Xva = add_bias(X_val,   self.bias) if self.bias else X_val

        Ttr = self._new_answers(t_train)
        Tva = self._new_answers(t_val) 

        N, M = Xtr.shape #N antall punkter, M antall features
        K = len(self.classes_) #antall klasser
        weights = np.zeros((M, K))

        best = np.inf
        no_improve = 0
        self.n_epochs_trained = 0
        self.loss_history = []


        for _ in range(epochs):
            # forward
            probabilities = _softmax(Xtr @ weights)
            # loss categorical cross-entropy
            loss = -np.mean(np.sum(Ttr * np.log(probabilities + 1e-12), axis=1)) #+1e-12 for å unngå log(0)
            self.loss_history.append(loss)

            grad = (Xtr.T @ (probabilities - Ttr)) / N  # gradient
            
            weights -= lr * grad#opppdater vekter

            # validering og tidlig stop
            Pv = _softmax(Xva @ weights)
            vloss = -np.mean(np.sum(Tva * np.log(Pv + 1e-12), axis=1))

            self.n_epochs_trained += 1
            if vloss + tol < best:
                best = vloss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= n_epochs_no_update:
                    break

        self.weights = weights  # lagre vektene

    def predict_proba(self, X):
        Xb = add_bias(X, self.bias) if self.bias else X
        return _softmax(Xb @ self.weights)  

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]



#MLP
def logistic(x): #sigmoid
    x = np.clip(x, -50, 50)
    return 1/(1+np.exp(-x))

def logistic_diff(y):
    return y * (1 - y)

class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.bias = bias
        self.dim_hidden = dim_hidden
        self.activ = logistic
        self.activ_diff = logistic_diff

        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.n_epochs_trained = 0
       
    def forward(self, X):
        """ 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        Z1=X@self.weights1
        sig = self.activ(Z1)
        sig = add_bias(sig, self.bias)
        Z2 = sig@self.weights2
        y = self.activ(Z2)
        return sig, y
        # return hidden_outs, outputs
    
    def fit(self, X_train, t_train, lr=0.001, epochs=100,
            X_val=None, t_val=None, tol=1e-4, n_epochs_no_update=10):
        """Initialize the weights. Train *epochs* many epochs.
        
        X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        lr is the learning rate
        """
        self.lr = lr
        
        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)

        if X_val is not None and t_val is not None:
            X_val_bias = add_bias(X_val, self.bias)
            T_val = t_val.reshape(-1,1)
        else:
            X_val_bias = None
            T_val = None

        
        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.n_epochs_trained = 0
        best_loss = float("inf")
        no_improve = 0
        
        for _ in range(epochs):
            # One epoch
            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)
            # The delta term on the output node:
            out_deltas = (outputs - T_train)
            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  

            # Update the weights:
            self.weights2 -= self.lr * hidden_outs.T @ out_deltas
            self.weights1 -= self.lr * X_train_bias.T @ hiddenact_deltas

            
            train_loss = -np.mean(T_train * np.log(outputs + 1e-12) + (1 - T_train) * np.log(1 - outputs + 1e-12)) #binary cross-entropy
            train_acc  = accuracy(self.predict(X_train), t_train)
            self.loss_history.append(train_loss)
            self.acc_history.append(train_acc)

            # val-metrics hvis gitt
            if X_val_bias is not None:
                _, val_outputs = self.forward(X_val_bias)
                val_loss = -np.mean(T_val * np.log(val_outputs + 1e-12) + (1 - T_val) * np.log(1 - val_outputs + 1e-12)) #binary cross-entropy
                val_acc  = accuracy(self.predict(X_val), t_val)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                monitor = val_loss
            else:
                monitor = train_loss

            #tidlig stopping
            if monitor + tol < best_loss:
                best_loss = monitor
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= n_epochs_no_update:
                    self.n_epochs_trained += 1
                    break

            self.n_epochs_trained += 1

    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)
    
    def predict_probability(self, X):
        Xb = add_bias(X, self.bias) # samme som i predict
        _, out = self.forward(Xb) # out er sannsynligheter 
        return out[:, 0] #returnerer uten bias         

    


#task multi-class neural network
#kombinasjon av OvR og mlp
class MLPOvR(NumpyClassifier):
    """
    Fungerer likt som OvR me logistisk regresjon. Her bruker jeg bare MLP istedenfor. kanskje den ikke hadde vært like efektiv som softmax mlp.
    """
    def __init__(self, bias=-1, dim_hidden=16):
        self.bias = bias
        self.dim_hidden = dim_hidden
        self.clfs = []
        self.classes_ = None

    def fit(self, X, y, lr=0.01, epochs=600,
            X_val=None, t_val=None, tol=1e-4, n_epochs_no_update=10):
        # lag én binær-MLP per klasse 
        self.classes_ = np.unique(y)
        self.clfs = []
        for k in self.classes_: #lager en mlp classifier per klasse
            t_bin = (y == k).astype(int) #setter alle klasser det ikke gjelder til 0 og klassen det gjelder til 1 i arrayet
            clf = MLPBinaryLinRegClass(bias=self.bias, dim_hidden=self.dim_hidden)
            clf.fit(X, t_bin,
                    lr=lr, epochs=epochs,
                    X_val=X_val, t_val=(t_val == k).astype(int) if t_val is not None else None,
                    tol=tol, n_epochs_no_update=n_epochs_no_update)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        # sannsynlighet per klasse (N,K)
        P = np.column_stack([clf.predict_probability(X) for clf in self.clfs])
        return P

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]




#Task tuning
cl = NumpyLinRegClass()
cl.fit(X_train, t2_train, 0.01, 200)
print("Accuracy on the validation set:", accuracy(cl.predict(X_val), t2_val))

lrs = [0.001, 0.01, 0.1, 0.5, 1.0]
epochs_list = [10, 50, 100, 200]

# løkke for å kunne sammenligne forskjellige lr og epochs
for lr in lrs:
    for ep in epochs_list:
        clf = NumpyLinRegClass(bias=1)
        clf.fit(X_train, t2_train, lr=lr, epochs=ep)
        val_acc = accuracy(clf.predict(X_val), t2_val)
        print("lr =", lr, "epochs =", ep, "val_acc =", val_acc) #logging


#Task scaling
scaler = StandardScaler()   
X_train_scaled = scaler.fit_transform(X_train) # finn mean/std og skaler treningsdata
X_val_scaled = scaler.transform(X_val) # bruk samme skalering på val-data
X_test_scaled = scaler.transform(X_test)# bruk samme skalering på test-data

cl = NumpyLinRegClass()
cl.fit(X_train_scaled, t2_train, 0.1, 100)
print("Accuracy on the validation set afte scaling:", accuracy(cl.predict(X_val_scaled), t2_val))

lrs = [0.001, 0.01, 0.1, 0.5, 1.0]
epochs_list = [10, 50, 100, 200]

# løkke for å kunne sammenligne forskjellige lr og epochs
for lr in lrs:
    for ep in epochs_list:
        clf = NumpyLinRegClass(bias=1)
        clf.fit(X_train_scaled, t2_train, lr=lr, epochs=ep)
        val_acc = accuracy(clf.predict(X_val_scaled), t2_val)
        #print("lr =", lr, "epochs =", ep, "val_acc =", val_acc)
        
plot_decision_regions(X_train, t2_train, cl)

#beste logistiske regresjon
best_logReg = find_the_best(
    X_train, t2_train, X_val, t2_val,
    lrs=[0.001, 0.01, 0.1, 0.3],
    tols=[0.00001, 0.0001, 0.001],
    early_stops=[10],
    max_epochs_list=[100, 200],
    mode="binary"
)
print("\nBest LogReg:",
      "scaled:", best_logReg["scaled"],
      "lr:", best_logReg["lr"],
      "tol:", best_logReg["tol"],
      "early_stop:", best_logReg["early_stop"],
      "max_ep:", best_logReg["max_ep"],
      "trained_ep:", best_logReg["trained_ep"],
      "val_loss:", best_logReg["val_loss"],
      "val_acc:", best_logReg["val_acc"])

#plot for loss sammenlignin
plt.figure(figsize=(6,4))
plt.plot(best_logReg["model"].loss_history, label="train loss")
if best_logReg["model"].val_loss_history:
    plt.plot(best_logReg["model"].val_loss_history, label="val loss")
plt.xlabel("epoch")
plt.ylabel("BCE loss")
plt.title("Loss vs epoch (best model)")
plt.legend()
plt.show()

#plot for sammenligning av accuracy
plt.figure(figsize=(6,4))
plt.plot(best_logReg["model"].acc_history, label="train acc")
if best_logReg["model"].val_acc_history:
    plt.plot(best_logReg["model"].val_acc_history, label="val acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy vs epoch (best model)")
plt.legend()
plt.show()

#multiclass OvR
best_ovr = find_the_best(
    X_train, t_multi_train, X_val, t_multi_val,
    lrs=[0.001, 0.01, 0.1, 0.3],
    tols=[0.00001, 0.0001, 0.001],
    early_stops=[10],
    max_epochs_list=[100, 200],
    mode="ovr"
)

print("\nBeste OvR:",
      "scaled:", best_ovr["scaled"],
      "lr:", best_ovr["lr"],
      "tol:", best_ovr["tol"],
      "early_stop:", best_ovr["early_stop"],
      "max_ep:", best_ovr["max_ep"],
      "val_acc:", best_ovr["val_acc"])

scaler = best_ovr["scaler"]
Xva = scaler.transform(X_val) if scaler is not None else X_val
Xte = scaler.transform(X_test) if scaler is not None else X_test
m = best_ovr["model"]

print("Val accuracy OvR:", m.prediction_accuracy(Xva, t_multi_val))
print("Test accuracy OvR:",        m.prediction_accuracy(Xte, t_multi_test))

Xplot = scaler.transform(X_train) if scaler is not None else X_train
plot_decision_regions(Xplot, t_multi_train, m)

#sammenliigning av OvR og softmax
best_soft = find_the_best(
    X_train, t_multi_train, X_val, t_multi_val,
    lrs=[0.001, 0.01, 0.1, 0.3],
    tols=[1e-5, 1e-4, 1e-3],
    early_stops=[10],
    max_epochs_list=[100, 200],
    mode="softmax"
)

print("\nSammenligning mellom OvR og Softmax:")
print("OvR - val_acc:", best_ovr["val_acc"], "val_loss:", best_ovr["val_loss"])
print("Softmax - val_acc:", best_soft["val_acc"], "val_loss:", best_soft["val_loss"])

print("Forskjell i accuracy:", best_soft["val_acc"] - best_ovr["val_acc"])
print("Forskjell i loss:", best_soft["val_loss"] - best_ovr["val_loss"])

#MLP skalering mot uten skalering
# uten skalering 
best_mlp_unscaled = find_the_best(
    X_train, t2_train, X_val, t2_val,
    lrs=[0.001, 0.01, 0.03],
    max_epochs_list=[500, 1000],
    use_scalings=(False,),
    mode="mlp",
    hidden_list=(6, 12),
    tols=(1e-5, 1e-4, 1e-3),
    early_stops=(10,)
)

#  med skalering 
best_mlp_scaled = find_the_best(
    X_train, t2_train, X_val, t2_val,
    lrs=[0.001, 0.01, 0.03],
    max_epochs_list=[500, 1000],
    use_scalings=(True,),
    mode="mlp",
    hidden_list=(6, 12),
    tols=(1e-5, 1e-4, 1e-3),
    early_stops=(10,)
)


print("\nBest (scaled):",
      "scaled", best_mlp_scaled["scaled"],
      "lr:", best_mlp_scaled["lr"],
      "tol:", best_mlp_scaled["tol"],
      "dim_hidden:", best_mlp_scaled["dim_hidden"],
      "epochs:", best_mlp_scaled["max_ep"],
      "trained_ep:", best_mlp_scaled["trained_ep"],
      "val_loss:", best_mlp_scaled["val_loss"],
      "val_acc:", best_mlp_scaled["val_acc"])

print("Best (unscaled):",
      "scaled:", best_mlp_unscaled["scaled"],
      "lr:", best_mlp_unscaled["lr"],
      "tol:", best_mlp_unscaled["tol"],
      "dim_hidden:", best_mlp_unscaled["dim_hidden"],
      "epochs:", best_mlp_unscaled["max_ep"],
      "trained_ep:", best_mlp_unscaled["trained_ep"],
      "val_loss:", best_mlp_unscaled["val_loss"],
      "val_acc:", best_mlp_unscaled["val_acc"])

scaler = best_mlp_scaled["scaler"]
Xtr_plot = scaler.transform(X_train) if scaler else X_train
plot_decision_regions(Xtr_plot, t2_train, best_mlp_scaled["model"])

#MLP treningsloss sammenlignet med validerings loss
plt.figure(figsize=(6,4))
plt.plot(best_mlp_scaled["model"].loss_history, label="Train loss")
plt.plot(best_mlp_scaled["model"].val_loss_history, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs (MLP)")
plt.legend()
plt.show()

#MLP treningsaccuracy mot validation accuracy
plt.figure(figsize=(6,4))
plt.plot(best_mlp_scaled["model"].acc_history, label="Train accuracy")
plt.plot(best_mlp_scaled["model"].val_acc_history, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs (MLP)")
plt.legend()
plt.show()


#se forskjell på MLP selvom de har samme hyperparametre fordi startvekter er forskjellige
n_runs = 10
accuracies = []

# hent beste verdier
lr = best_mlp_scaled["lr"]
epochs = best_mlp_scaled["max_ep"]
dim_hidden = best_mlp_scaled["dim_hidden"]
scaler = best_mlp_scaled["scaler"]
tol_ = best_mlp_scaled["tol"]
n_epochs_no_update_=best_mlp_scaled["early_stop"]

# skaler data hvis nødvendig
Xtr = scaler.transform(X_train) if scaler else X_train
Xva = scaler.transform(X_val) if scaler else X_val

for i in range(n_runs):
    clf = MLPBinaryLinRegClass(bias=-1, dim_hidden=dim_hidden)
    clf.fit(Xtr, t2_train, lr=lr, epochs=epochs,
            X_val=Xva, t_val=t2_val, tol=tol_, n_epochs_no_update=n_epochs_no_update_)
    acc = accuracy(clf.predict(Xva), t2_val)
    accuracies.append(acc)
    print(f"Run {i+1}: val_acc = {acc:.4f}")

# beregn gjennomsnitt og standardavvik
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print("\nMean accuracy over 10 runs:", round(mean_acc, 4))
print("Standard deviation:", round(std_acc, 4))

#  OvR-mlp
best_ovr_mlp = find_the_best(
    X_train, t_multi_train, X_val, t_multi_val,
    lrs=[0.001, 0.01, 0.03],
    max_epochs_list=[500, 1000],
    use_scalings=(True,),
    mode="ovr_mlp",
    hidden_list=(6, 12),
    tols=(1e-5, 1e-4, 1e-3),
    early_stops=(10,)
)

print("\nBest OvR-MLP (scaled):",
      "scaled:", best_ovr_mlp["scaled"],
      "lr:", best_ovr_mlp["lr"],
      "tol:", best_ovr_mlp["tol"],
      "dim_hidden:", best_ovr_mlp["dim_hidden"],
      "epochs:", best_ovr_mlp["max_ep"],
      "trained_ep:", best_ovr_mlp["trained_ep"],
      "val_loss:", best_ovr_mlp["val_loss"],
      "val_acc:", best_ovr_mlp["val_acc"],"\n")

scaler = best_ovr_mlp["scaler"]
Xtr_plot = scaler.transform(X_train) if scaler else X_train
plot_decision_regions(Xtr_plot, t_multi_train, best_ovr_mlp["model"])




def maybe_scale_set(X, best_dict):
    scaled = best_dict.get("scaled", False)
    scaler = best_dict.get("scaler", None)
    return scaler.transform(X) if (scaled and scaler is not None) else X

# accuracy for lineær regresjon classifier
lin_train = accuracy(cl.predict(X_train_scaled), t2_train)
lin_val   = accuracy(cl.predict(X_val_scaled),   t2_val)
lin_test  = accuracy(cl.predict(X_test_scaled),  t2_test)

# beste logistiske regresjon 
log_info  = best_logReg
log_model = best_logReg["model"]

Xtr_log = maybe_scale_set(X_train, log_info)
Xva_log = maybe_scale_set(X_val,   log_info)
Xte_log = maybe_scale_set(X_test,  log_info)

log_train = accuracy(log_model.predict(Xtr_log), t2_train)
log_val   = accuracy(log_model.predict(Xva_log), t2_val)
log_test  = accuracy(log_model.predict(Xte_log), t2_test)

#MLP skalert
mlp_info  = {"scaled": True, "scaler": best_mlp_scaled.get("scaler", None)}
mlp_model = best_mlp_scaled["model"]

Xtr_mlp = maybe_scale_set(X_train, mlp_info)
Xva_mlp = maybe_scale_set(X_val,   mlp_info)
Xte_mlp = maybe_scale_set(X_test,  mlp_info)

mlp_train = accuracy(mlp_model.predict(Xtr_mlp), t2_train)
mlp_val   = accuracy(mlp_model.predict(Xva_mlp), t2_val)
mlp_test  = accuracy(mlp_model.predict(Xte_mlp), t2_test)

#bruker pandas for fin utskrift
data = {
    "training":   [lin_train,  log_train,  mlp_train],
    "validation": [lin_val,    log_val,    mlp_val],
    "test":       [lin_test,   log_test,   mlp_test],
}
index = ["lineær regresjon", "logistisk regresjon", "MLP"]

df = pd.DataFrame(data, index=index).round(4)
print(df) 

for name, model in [("Linear", cl), ("Logistic", log_model), ("MLP", mlp_model)]:
    y_pred = model.predict(X_test_scaled)
    prec = precision_score(t2_test, y_pred)
    rec = recall_score(t2_test, y_pred)
    print(name + ": precision=" + str(round(prec, 3)) + ", recall=" + str(round(rec, 3)))


#samme med multiclass classifiers
# OvR logistisk
ovr_model = best_ovr["model"]
Xtr_ovr = maybe_scale_set(X_train, best_ovr)
Xva_ovr = maybe_scale_set(X_val,   best_ovr)
Xte_ovr = maybe_scale_set(X_test,  best_ovr)

ovr_train = np.mean(ovr_model.predict(Xtr_ovr) == t_multi_train)
ovr_val   = np.mean(ovr_model.predict(Xva_ovr) == t_multi_val)
ovr_test  = np.mean(ovr_model.predict(Xte_ovr) == t_multi_test)

# Softmax
soft_model = best_soft["model"]
Xtr_soft = maybe_scale_set(X_train, best_soft)
Xva_soft = maybe_scale_set(X_val,   best_soft)
Xte_soft = maybe_scale_set(X_test,  best_soft)

soft_train = np.mean(soft_model.predict(Xtr_soft) == t_multi_train)
soft_val   = np.mean(soft_model.predict(Xva_soft) == t_multi_val)
soft_test  = np.mean(soft_model.predict(Xte_soft) == t_multi_test)

# OvR-MLP
ovr_mlp_model = best_ovr_mlp["model"]
Xtr_ovrmlp = maybe_scale_set(X_train, best_ovr_mlp)
Xva_ovrmlp = maybe_scale_set(X_val,   best_ovr_mlp)
Xte_ovrmlp = maybe_scale_set(X_test,  best_ovr_mlp)

ovrmlp_train = np.mean(ovr_mlp_model.predict(Xtr_ovrmlp) == t_multi_train)
ovrmlp_val   = np.mean(ovr_mlp_model.predict(Xva_ovrmlp) == t_multi_val)
ovrmlp_test  = np.mean(ovr_mlp_model.predict(Xte_ovrmlp) == t_multi_test)


data_mc = {
    "training":   [ovr_train,  soft_train,  ovrmlp_train],
    "validation": [ovr_val,    soft_val,    ovrmlp_val],
    "test":       [ovr_test,   soft_test,   ovrmlp_test],
}
index_mc = ["OvR (logreg)", "Softmax (logreg)", "OvR-MLP"]

df_mc = pd.DataFrame(data_mc, index=index_mc).round(4)
print(df_mc)          

