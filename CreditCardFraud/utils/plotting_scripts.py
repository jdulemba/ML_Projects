from pdb import set_trace

# Set Matplotlib defaults
from matplotlib import pyplot as plt
from utils.styles import style_dict
plt.style.use(style_dict)

import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

def plot_confusion_matrix(df, fig_title="Confusion Matrix"):
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
        fig.suptitle(fig_title, y=title_loc)
    else:
        fig.suptitle(fig_title)

    return fig


def plot_roc(df, fig_title="ROC Curve", fpr_thresh=None):
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
    ncols = int(np.ceil(nclassifiers/max_labels_per_axis))
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

        if fpr_thresh is None: axs[-1].plot([0, 1], [0, 1], color="k", lw=2, linestyle="--")
        axs[-1].set_xlim(0.0, 1.0 if fpr_thresh is None else fpr_thresh)
        axs[-1].set_ylim(0.0, 1.05)
        axs[-1].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        axs[-1].legend(loc="lower right", fontsize=10)
        axs[-1].set_box_aspect(1)

    # rescale figure height to remove white space
    if ncols * nrows > 1:
        ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
        fig.set_figheight(ax_max_height*fig_height)
        fig.suptitle(fig_title, y=0.9 if nrows == 1 else 1.04)
    else:
        fig.suptitle(fig_title)

    return fig


def plot_precision_recall(df, fig_title="Precision-Recall Curve"):
    max_labels_per_axis = 6
    nclassifiers = len(df)
    ncols = int(np.ceil(nclassifiers/max_labels_per_axis))
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
        fig.suptitle(fig_title, y=0.9 if nrows == 1 else 1.04)
    else:
        fig.suptitle(fig_title)

    return fig



def plot_df(df, fig_title="Results"):
    # determine number of rows and columns to be produced based on the size of the df
    max_labels_per_axis = 6
    nclass = df.shape[-1]
    ncols = int(np.ceil(nclass/max_labels_per_axis))
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
        fig.suptitle(fig_title, y=0.9 if nrows == 1 else 1.04)
    else:
        fig.suptitle(fig_title)

    return fig

def plot_optimization_results(classifier, class_type=""):
    # determine number of rows and columns to be produced based on the size of the df
    max_bins_per_axis = 30
    results = ["mean_train_score", "std_train_score", "mean_test_score", "std_test_score"]
    df = pd.DataFrame(classifier.cv_results_, columns=results)
    nclass = df.shape[0]
    ncols = 1
    nrows = int(np.ceil(nclass/max_bins_per_axis))
    bins_per_axis = int(np.ceil(nclass/nrows))

    max_nrows = 5
    if nrows > max_nrows:
        nfigs = int(np.ceil(nrows/max_nrows))
            #split DataFrame into chunks based on number of figs
        df_fig_list = [df.iloc[i:i+max_nrows*bins_per_axis, :] for i in range(0, len(df), max_nrows*bins_per_axis)]
        figs = []
        for fig_idx in range(len(df_fig_list)): 
                # split dataframe into more chunks for each figure
            df_list = [df_fig_list[fig_idx].iloc[i:i+bins_per_axis, :] for i in range(0, len(df_fig_list[fig_idx]), bins_per_axis)]
            fig = plt.figure(constrained_layout=True, figsize=(15.0, 10.0))
            gs = gridspec.GridSpec(ncols=ncols, nrows=len(df_list), figure=fig)
            axs = []

            for idx, df_chunk in enumerate(df_list):
                axs.append(fig.add_subplot(gs[idx]))

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

            #set_trace()
            axs[0].legend(loc="upper right", fontsize=12)
            axs[0].set_ylabel(f"{classifier.scoring.capitalize()} Score", size=16)
            axs[-1].set_xlabel("Hyperparameter Combination", size=16)

            # rescale figure height to remove white space
            if ncols * nrows > 1:
                ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
                fig.set_figheight(ax_max_height*fig_height)
                fig.suptitle(f"{class_type} Validation Curves", y=0.9 if nrows == 1 else 1.04)
            else:
                fig.suptitle(f"{class_type} Validation Curves")

            figs.append(fig)

        return figs

    else:
        fig = plt.figure(constrained_layout=True, figsize=(15.0, 10.0))
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
        axs = []
            #split DataFrame into chunks
        df_list = [df.iloc[i:i+bins_per_axis, :] for i in range(0, len(df), bins_per_axis)]
        for idx, df_chunk in enumerate(df_list):
            axs.append(fig.add_subplot(gs[idx]))

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

        axs[0].legend(loc="upper right", fontsize=12)
        axs[0].set_ylabel(f"{classifier.scoring.capitalize()} Score", size=16)
        axs[-1].set_xlabel("Hyperparameter Combination", size=16)

        # rescale figure height to remove white space
        if ncols * nrows > 1:
            ax_max_height, fig_height = max([ax.get_position().ymax for ax in axs]), fig.get_size_inches()[-1]
            fig.set_figheight(ax_max_height*fig_height)
            fig.suptitle(f"{class_type} Validation Curves", y=0.9 if nrows == 1 else 1.04)
        else:
            fig.suptitle(f"{class_type} Validation Curves")

        return fig


def plot_GAsearch_results(clf, plot_type=""):
    supported_plot_types = ["FitnessEvolution", "SearchSpace"]
    assert plot_type in supported_plot_types, f"{plot_type} is not a valid type of plot to make, must be {supported_plot_types}."

    if plot_type == "FitnessEvolution":
        #import seaborn as sns
        metric = "fitness"
        fitness_vals, fitness_std = np.array(clf.history[metric]), np.array(clf.history[f"{metric}_std"])

        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax.plot(range(len(clf)), fitness_vals, color="b")
        ax.fill_between(
            range(len(clf)),
            fitness_vals - fitness_std,
            fitness_vals + fitness_std,
            alpha=0.2, color="b", lw=2
        )
        ax.set_title(f"{metric.capitalize()} Average Evolution Over Generations", size=20)
        ax.set_xlabel("Generations", size=16)
        ax.set_ylabel(f"{clf.scoring.upper()} {clf.refit_metric.capitalize()}", size=16)


    if plot_type == "SearchSpace":
        import seaborn as sns
        sns.set_style(style_dict)

        from sklearn_genetic.utils import logbook_to_pandas
        from sklearn_genetic.genetic_search import GAFeatureSelectionCV
        height, s = 2, 25
        features = None
        """
        Parameters
        ----------
        clf: clf object
            A fitted clf from :class:`~sklearn_genetic.GASearchCV`
        height: float, default=2
            Height of each facet
        s: float, default=5
            Size of the markers in scatter plot
        features: list, default=None
            Subset of features to plot, if ``None`` it plots all the features by default

        Returns
        -------
        Pair plot of the used hyperparameters during the search

        """

        if isinstance(clf, GAFeatureSelectionCV):
            raise TypeError(
                "Estimator must be a GASearchCV instance, not a GAFeatureSelectionCV instance"
            )


        df = logbook_to_pandas(clf.logbook)
        if features:
            stats = df[features].astype(np.float64)
        else:
            variables = [*clf.space.parameters, clf.refit_metric]
            stats = df[variables].select_dtypes(["number", "bool"]).astype(np.float64)

        grid = sns.PairGrid(stats, diag_sharey=False, height=height)
        grid = grid.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)
        grid = grid.map_lower(
            sns.kdeplot,
            fill=True,
            cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
        )
        grid = grid.map_diag(sns.kdeplot, fill=True, alpha=0.2, color="red")

        [ax.tick_params(axis="both", which="both", left=False, right=False, top=False, bottom=False) for ax in grid.axes.ravel()]
        fig = grid.figure
        ax_max_height, fig_height = max([ax.get_position().ymax for ax in grid.axes.ravel()]), fig.get_size_inches()[-1]
        fig.set_figheight(ax_max_height*fig_height)
        fig.suptitle("Hyperparameter Search Space", y=1.04, size=20)

    return fig
