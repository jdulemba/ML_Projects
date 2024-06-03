from pdb import set_trace

# Set Matplotlib defaults
from matplotlib import pyplot as plt
from utils.styles import style_dict
plt.style.use(style_dict)

import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

def plot_confusion_matrix(df, data_type=""):
    import seaborn as sns

    class_names = { 0 : "Not Fraud", 1 : "Fraud"}
    nclassifiers = len(df)
    max_ncols = 4
    if nclassifiers <= 3:
        nrows = 1
        ncols = nclassifiers
    elif nclassifiers == 4:
        nrows, ncols = 2, 2
    else:
        nrows = int(np.ceil(nclassifiers/max_ncols))
        ncols = int(np.ceil(nclassifiers/nrows))

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    axs = []
    for idx, key in enumerate(df.index):
        cm_df = pd.DataFrame(df.loc[key, :].values[0], index=class_names.values(), columns=class_names.values())
    
            # plot confusion matrix to visualize how correct the model is at predicting fraud/not fraud
        axs.append(fig.add_subplot(gs[idx]))
        sns.heatmap(cm_df, annot=True, cbar=None, cmap="Blues", fmt="g", ax=axs[idx], annot_kws={"size": 12})
        axs[idx].set(xlabel="Predicted Class", ylabel="True Class")
        axs[idx].set_title(key, size=10 if nclassifiers > 2 else 20)
        axs[idx].set_box_aspect(1)
        axs[idx].grid(False)
        axs[idx].tick_params(axis="both", which="both", left=False, right=False, top=False, bottom=False)

    # rescale figure height to remove white space
    if ncols * nrows > 1:
        ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
        fig.set_figheight(ax_max_height*fig_height)
        if nrows == 1:
            title_loc = 0.9
        else:
            title_loc = 1.04
        fig.suptitle(f"Confusion Matrix for {data_type} Data", y=title_loc)
    else:
        fig.suptitle(f"Confusion Matrix for {data_type} Data")

    return fig


def plot_roc(df, fpr_thresh=None):
    from sklearn.metrics import auc
    def partial_auc(fpr, tpr, max_fpr):
        "Taken from here https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/metrics/_ranking.py#L350-L356"
        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
    
        return auc(fpr, tpr)

    max_labels_per_axis = 6
    nclassifiers = len(df)
    ncols = int(np.ceil(nclassifiers/max_labels_per_axis)) if nclassifiers < 16 else int(np.ceil(nclassifiers/4))
    nrows = 1
    labels_per_axis = int(np.ceil(nclassifiers/ncols))
    if ncols >= 4:
        ncols, nrows = 2, 2

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    axs = []

        #split DataFrame into chunks
    df_list = [df.iloc[i:i+labels_per_axis, :] for i in range(0, nclassifiers, labels_per_axis)]
    for idx, df_chunk in enumerate(df_list):
        axs.append(fig.add_subplot(gs[idx]))
        # plot curve for each classifier
        for class_name in df_chunk.index:
            fpr, tpr, thresh = df_chunk.loc[class_name, ["ROC_FPR", "ROC_TPR", "ROC_Thresh"]]
            roc_auc = auc(fpr, tpr)
            if fpr_thresh is not None:
                roc_auc = partial_auc(fpr, tpr, max_fpr=fpr_thresh)

            axs[-1].plot(fpr, tpr,lw=2, label=class_name+ " (AUC=%0.3f)" % roc_auc if fpr_thresh is None else class_name+ " (AUCx100=%0.3f)" % (roc_auc*100))

        axs[-1].plot([0, 1 if fpr_thresh is None else fpr_thresh], [0, 1], color="k", lw=2, linestyle="--")
        axs[-1].set_xlim(0.0, 1.0 if fpr_thresh is None else fpr_thresh)
        axs[-1].set_ylim(0.0, 1.05)
        axs[-1].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        axs[-1].legend(loc="lower right", fontsize=10)
        axs[-1].set_box_aspect(1)

    # rescale figure height to remove white space
    if ncols * nrows > 1:
        ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
        fig.set_figheight(ax_max_height*fig_height)
        fig.suptitle("ROC Curve", y=0.9 if nrows == 1 else 1.04)
    else:
        fig.suptitle("ROC Curve")

    return fig


def plot_precision_recall(df):
    max_labels_per_axis = 6
    nclassifiers = len(df)
    ncols = int(np.ceil(nclassifiers/max_labels_per_axis)) if nclassifiers < 16 else int(np.ceil(nclassifiers/4))
    nrows = 1
    labels_per_axis = int(np.ceil(nclassifiers/ncols))
    if ncols >= 4:
        ncols, nrows = 2, 2

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    axs = []

        #split DataFrame into chunks
    df_list = [df.iloc[i:i+labels_per_axis, :] for i in range(0, nclassifiers, labels_per_axis)]
    for idx, df_chunk in enumerate(df_list):
        axs.append(fig.add_subplot(gs[idx]))
        for class_name in df_chunk.index:
            precision, recall, thresh, avg_prec = df_chunk.loc[class_name, :]
            axs[-1].plot(recall, precision,lw=2, label=class_name+ " (AP=%0.3f)" % avg_prec)
    
        axs[-1].set_xlim(0.0, 1.0)
        axs[-1].set_ylim(0.0, 1.05)
        axs[-1].set(xlabel="Recall", ylabel="Precision")
        axs[-1].legend(loc="lower left", fontsize=10)
        axs[-1].set_box_aspect(1)

    # rescale figure height to remove white space
    if ncols * nrows > 1:
        ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
        fig.set_figheight(ax_max_height*fig_height)
        fig.suptitle("Precision-Recall Curve", y=0.9 if nrows == 1 else 1.04)
    else:
        fig.suptitle("Precision-Recall Curve")

    return fig



def plot_df(df, data_type=""):
    # determine number of rows and columns to be produced based on the size of the df
    max_labels_per_axis = 6
    nclass = df.shape[-1]
    ncols = int(np.ceil(nclass/max_labels_per_axis)) if nclass < 16 else int(np.ceil(nclass/4))
    nrows = 1
    labels_per_axis = int(np.ceil(nclass/ncols))
    if ncols >= 4:
        ncols, nrows = 2, 2

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    axs = []

        #split DataFrame into chunks
    df_list = [df.iloc[:, i:i+labels_per_axis] for i in range(0, len(df.transpose()), labels_per_axis)]
    for idx, pd_df in enumerate(df_list):
        axs.append(fig.add_subplot(gs[idx]))
        pd_df.plot(ax=axs[-1])
        axs[-1].set(xlabel="Score Type", ylabel="Score Value")
        axs[-1].legend(loc="lower left", fontsize=10)
        axs[-1].set_box_aspect(1)

    # rescale figure height to remove white space
    if ncols * nrows > 1:
        ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
        fig.set_figheight(ax_max_height*fig_height)
        fig.suptitle(f"{data_type} Results", y=0.9 if nrows == 1 else 1.04)
    else:
        fig.suptitle(f"{data_type} Results")

    return fig

def plot_gridsearch_results(classifier, class_type=""):
    # determine number of rows and columns to be produced based on the size of the df
    max_bins_per_axis = 30
    results = ["mean_train_score", "std_train_score", "mean_test_score", "std_test_score"]
    df = pd.DataFrame(classifier.cv_results_, columns=results)
    nclass = df.shape[0]
    ncols = 1
    nrows = int(np.ceil(nclass/max_bins_per_axis))
    bins_per_axis = int(np.ceil(nclass/nrows))


    fig = plt.figure(constrained_layout=True, figsize=(15.0, 10.0))
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    axs = []
        #split DataFrame into chunks
    df_list = [df.iloc[i:i+bins_per_axis, :] for i in range(0, len(df), bins_per_axis)]
    for idx, df_chunk in enumerate(df_list):
        axs.append(fig.add_subplot(gs[idx]))
        axs[-1].set(ylabel=f"{classifier.scoring} Score")

        axs[-1].plot(df_chunk["mean_train_score"], label="Training", color="darkorange", lw=2)
        axs[-1].fill_between(
            df_chunk.index.values,
            df_chunk["mean_train_score"] - df_chunk["std_train_score"],
            df_chunk["mean_train_score"] + df_chunk["std_train_score"],
            alpha=0.2, color="darkorange", lw=2
        )

        axs[-1].plot(df_chunk["mean_test_score"], label="Cross-Validation", color="navy", lw=2)
        axs[-1].fill_between(
            df_chunk.index.values,
            df_chunk["mean_test_score"] - df_chunk["std_test_score"],
            df_chunk["mean_test_score"] + df_chunk["std_test_score"],
            alpha=0.2, color="navy", lw=2
        )
        axs[-1].set_xlim(df_chunk.index.values[0], df_chunk.index.values[-1])

    axs[0].legend(loc="upper right", fontsize=10)
    axs[-1].set(xlabel="Gridsearch Combination")

    # rescale figure height to remove white space
    if ncols * nrows > 1:
        ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
        fig.set_figheight(ax_max_height*fig_height)
        fig.suptitle(f"{class_type} Validation Curves", y=0.9 if nrows == 1 else 1.04)
    else:
        fig.suptitle(f"{class_type} Validation Curves")

    return fig
