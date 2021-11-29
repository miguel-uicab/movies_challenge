# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import re


def info_user(data=None):
    total_user = data.userId.value_counts().to_frame('count').reset_index()
    total_user.columns = ['userId', 'count']
    print('Descriptivos de las frecuencias de revisiones de películas hechas '
          'por los usuarios.')
    return total_user['count'].describe()


def clean_string(string=None):
    raw_string = re.sub(r"[ : ]", " ", string)
    # replace = (raw_string.replace("|", " ")
    #                       .replace("-", " ")
    #                       .replace("'", " "))
    replace = raw_string.replace("|", " ").replace("'", " ")
    clean_charac = re.sub(r"[\.\,\[\]\(\)\_\#\*\¢\$\&\:\;\·\%\|]",
                          "",
                          replace)
    clean_lower = clean_charac.lower()
    if clean_lower == 'no genres listed':
        output = [clean_lower]
    else:
        output = clean_lower.split()
        # output = [char for char in clean_lower]
    # clean_accents = unidecode(clean_lower)
    return output


def corrheatmep(data=None):
    "Construye el mapa de calor que informa del nivel de correlación lineal"
    "de las variables numéricas."

    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(method='spearman').abs(),
                square=True,
                annot=True,
                cmap='RdBu',
                vmin=-1,
                vmax=1)
    plt.show()


def correlated_variables_names(data, threshold=0.8):
    "Encuentra el nombre de variables numéricas altamente correlacionadas"
    "con otras, dado un umbral."

    corr_matrix = (data.corr(method='spearman')
                       .abs())
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                      k=1).astype(np.bool))
    names = [column for column in upper.columns if any(upper[column] > threshold)]

    return names


def PCA_function(data):
    "Exhibe posibles valores outliers multidimensionales de "
    "variables numéricas usando Análisis de Componentes Principales."

    dim_reduction = PCA()
    Xc = dim_reduction.fit_transform(scale(data))
    comp_columns = ['componente_' + str(j + 1) for j in range(data.shape[1])]
    componentes = pd.DataFrame(Xc, columns=comp_columns, index=data.index)
    # Porcentaje de resumen de las primeras tres componentes:
    print('Varianza explicada por la primera componente: %0.1f%%' % (
          sum(dim_reduction.explained_variance_ratio_[:1] * 100)))
    print('Varianza explicada por las primeras 2 componentes: %0.1f%%' % (
          sum(dim_reduction.explained_variance_ratio_[:2] * 100)))
    print('Varianza explicada por las primeras 3 componentes: %0.1f%%' % (
          sum(dim_reduction.explained_variance_ratio_[:3] * 100)))
    # Primera Componente VS Segunda Componente ####
    fig = px.scatter(componentes,
                     x=componentes.iloc[:, 0].name,
                     y=componentes.iloc[:, 1].name)
    fig.show()
    # Último Componente VS Penúltimo Componente ####
    fig1 = px.scatter(componentes,
                      x=componentes.iloc[:, -2].name,
                      y=componentes.iloc[:, -1].name)
    fig1.show()

    return componentes


def mahalanobis_visualization(database=None, cov=None, alpha=0.01,
                              threshold=None, visualization=False):
    _, p = database.shape
    "Devuelve los índices de las observaciones multidimensionales más "
    "alejadas del centro según la distancia de T2 de Mahalanobis. "

    def mahalanobis(data=None, cov=None):
        x_mu = data - np.mean(data)
        if not cov:
            cov = np.cov(data.values.T)
        inv_covmat = np.linalg.inv(cov)
        left = np.dot(x_mu, inv_covmat)
        mahal = np.dot(left, x_mu.T)

        return mahal.diagonal()

    chi_statistic = chi2.ppf(1 - alpha, df=p)

    database['mahalanobis'] = mahalanobis(data=database)

    if visualization:
        fig = px.scatter(y=database['mahalanobis'],
                         x=list(database.index))
        fig.add_hline(y=chi_statistic,
                      line_dash="dot",
                      annotation_text=f"chi square alpha={alpha}",
                      annotation_position="bottom right")
        fig.show()
    if not threshold:
        threshold = chi_statistic
    outliers_index = database[database.mahalanobis > threshold].index

    return outliers_index


def TwoSampleT2Test(X=None, Y=None, alpha=0.05):
    "Test Ji-Cuadrado no paramétrico (tamaño de data grande) para diferencia "
    "de medias de dos poblaciones."

    nx, p = X.shape
    ny, _ = Y.shape
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    Sx = np.cov(X, rowvar=False)
    Sy = np.cov(Y, rowvar=False)
    S_pooled = (Sx / nx) + (Sy / ny)
    T2 = np.matmul(np.matmul(delta.transpose(),
                             np.linalg.inv(S_pooled)),
                   delta)
    statistic = chi2.ppf(1 - alpha, df=p)
    p_value = 1 - chi2.cdf(T2, df=p)
    print(f"T2 value: {T2}")
    print(f"Chi-statistic: {statistic}")
    print(f"Degrees of freedom: {p}")
    print(f"p-value: {p_value}")


#  monotone_constraints='(-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,0,-1,1,1,1,0,0,1,0)',
def learning_curve(X_train=None, X_test=None,
                   y_train=None, y_test=None,
                   Pipeline=None, random_state=0,
                   n_estimators=100, reg_alpha=10,
                   learning_rate=0.3, max_depth=6,
                   subsample=1, min_child_weight=1,
                   gamma=0, eval_metric='auc'):
    "Construye una curva de aprendizaje para XGBClassifier."

    model = XGBClassifier(use_label_encoder=False,
                          random_state=random_state,
                          objective='binary:logistic',
                          booster='gbtree',
                          eval_metric=eval_metric,
                          tree_method='approx',
                          n_estimators=n_estimators,
                          reg_alpha=reg_alpha,
                          learning_rate=learning_rate,
                          max_depth=max_depth,
                          subsample=subsample,
                          min_child_weight=min_child_weight,
                          gamma=gamma)
    X_train_p = Pipeline.fit_transform(X_train)
    X_test_p = Pipeline.transform(X_test)
    evalset = [(X_train_p, y_train), (X_test_p, y_test)]
    model.fit(X_train_p,
              y_train,
              eval_metric=eval_metric,
              eval_set=evalset,
              verbose=False)
    results = model.evals_result()
    # plot learning curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, n_estimators + 1)),
                             y=results['validation_0'][eval_metric],
                             mode='lines+markers',
                             name='train',
                             marker_color='purple'))
    fig.add_trace(go.Scatter(x=list(range(1, n_estimators + 1)),
                             y=results['validation_1'][eval_metric],
                             mode='lines+markers',
                             name='test',
                             marker_color='orange'))
    fig.update_layout(showlegend=True,
                      title_text="Curva de aprendizaje",
                      xaxis_title='Número de estimadores',
                      yaxis_title=f'{eval_metric}')
    fig.show()


def curve_roc(clf=None, X_test=None, y_test=None):
    "Construye Curva-ROC dado un estimador ya entrenado."

    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    fig = px.area(x=fpr,
                  y=tpr,
                  title=f'ROC-Curve (AUC={roc_auc:.4f})',
                  labels=dict(x='False Positive Rate',
                              y='True Positive Rate'),
                  width=700,
                  height=500)
    fig.add_shape(type='line',
                  line=dict(dash='dash'),
                  x0=0,
                  x1=1,
                  y0=0,
                  y1=1)
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


def curve_pr(clf=None, X_test=None, y_test=None):
    "Construye Curva-PR dado un estimador ya entrenado."

    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    a_p_s = average_precision_score(y_test, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    fig = px.area(x=recall,
                  y=precision,
                  title=f'PR-Curve (AUC={a_p_s:.4f})',
                  labels=dict(x='Recall',
                              y='Precision'),
                  width=700,
                  height=500)
    fig.add_shape(type='line',
                  line=dict(dash='dash'),
                  x0=0,
                  x1=1,
                  y0=1,
                  y1=0)
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


def curve_calibration(clf=None, X_test=None, y_test=None):
    "Construye Curva de Calibración."

    probs = clf.predict_proba(X_test)[:, 1]
    fop, mpv = calibration_curve(y_test,
                                 probs,
                                 n_bins=50,
                                 normalize=True)
    fig = go.Figure(data=go.Scatter(x=mpv,
                                    y=fop,
                                    mode='lines+markers'))
    fig.add_shape(type='line',
                  line=dict(dash='dash'),
                  x0=0,
                  x1=1,
                  y0=0,
                  y1=1)
    fig.update_layout(title='Calibration Test',
                      xaxis_title='Mean Predicted Probability in each bin',
                      yaxis_title='Ratio of positives')
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=1)
    fig.update_xaxes(constrain='domain')
    # fig.update_traces(marker=dict(size=5,
    #                               line=dict(width=2,
    #                                         color='gray')),
    #                  line=dict(width=1,
    #                            color='orange'))
    fig.update_traces(marker=dict(size=6, color='gray'),
                      line=dict(width=1,
                                color='orange'))
    fig.show()
