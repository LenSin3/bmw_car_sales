from seaborn import palettes
from seaborn.utils import ci
import funcs
import utils
# import dependencies
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import wordcloud


def plot_qa(df_null):
    """Plot Count of Nulls of columns with nulls.

    The plot is done for columns with nulls in Dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe to check and plot for nulls.
        

    Returns
    -------
    Seaborn.barplot
    """
        
    null_columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsInColumn', 'PercentOfNullsInData']
    df_null_cols = df_null.columns

    # check if columns match
    mem_cols = all(col in null_columns for col in df_null_cols)
    if mem_cols:
        if df_null['CountOfNulls'].sum() == 0:
            print("There are zero nulls in the DataFrame.")
            print("No plot to display!")
        else:
            null_df = df_null.loc[df_null['CountOfNulls'] > 0]
            null_df.reset_index(drop = True, inplace = True)

            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 10)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            bar = sns.barplot(y = 'PercentOfNullsInColumn', x = 'Feature' , data = null_df, ci = False)
            for i in range(len(null_df)):
                bar.text(i, null_df['PercentOfNullsInColumn'][i] + 0.25, str(round(null_df['PercentOfNullsInColumn'][i], 2)),
                fontdict= dict(color = 'blue', fontsize = 10, horizontalalignment = 'center'))
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.title('Percentage of Nulls in Column')
            plt.show()
    else:
        raise utils.UnexpectedDataFrame(df_null)

def plot_qa_mtpltlib(df_null):
    """Plot Count of Nulls of columns with nulls.

    The plot is done for columns with nulls in Dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe to check and plot for nulls.
        

    Returns
    -------
    matplotlib.pyplot.barplot 
    """
        
    null_columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsInColumn', 'PercentOfNullsInData']
    df_null_cols = df_null.columns

    # check if columns match
    mem_cols = all(col in null_columns for col in df_null_cols)
    if mem_cols:
        if df_null['CountOfNulls'].sum() == 0:
            print("There are zero nulls in the DataFrame.")
            print("No plot to display!")
        else:
            null_df = df_null.loc[df_null['CountOfNulls'] > 0]
            null_df.reset_index(drop = True, inplace = True)

            plt.figure(figsize=(15, 8))
            plt.bar('Feature', 'CountOfNulls', data = df_null, color = 'orange', width = 0.9, align = 'center', edgecolor = 'blue')
            # i = 1.0
            # j = 2000
            # for i in range(len(null_df)):
            #     plt.annotate(null_df['PercentOfNullsInColumn'][i], (-0.1 + i, null_df['PercentOfNullsInColumn'][i] + j))
            plt.xticks(rotation = 90)
            plt.xlabel("Columns")
            plt.ylabel("Percentage")
            plt.title("Count of Nulls in Column")
            plt.show()
    else:
        raise utils.UnexpectedDataFrame(df_null)


def plot_unique_vals_count(df):
    """Plot count of unique values per column.

    The plot is done for unique values of all columns.

    Parameters
    ----------
    df : Dataframe
        Dataframe to check for unique values per column.        

    Returns
    -------
    seaborn.barplot
    """
    unique_vals = funcs.unique_vals_counts(df) 
    fig, ax = plt.subplots()
    # the size of A4 paper lanscape
    fig.set_size_inches(15, 12)
    sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
    bar = sns.barplot(y = 'count', x = 'column' , data = unique_vals)
    for i in range(len(unique_vals)):
                 bar.text(i, unique_vals['count'][i] + 0.25, str(unique_vals['count'][i]),
                 fontdict= dict(color = 'blue', fontsize = 10, horizontalalignment = 'center'))
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.title('Count of Unique Values per Column')
    plt.show()


def plot_unique_vals_column(df, col, normalize = False):
    """Plot value counts in a column.

    Value counts are calculated for a single column and plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to check for unique values
    normalized : bool, optional
        If true this function normalizes the counts.
        (Default value = False)

    Returns
    -------
    seaborn.barplot
    """
    if normalize:
            unique_col_vals = funcs.unique_vals_column(df, col, normalize = True) 
            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 12)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            sns.barplot(x = 'percentOfTotal', y = col , data = unique_col_vals)
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.title('Percentage of Unique Values in {}'.format(col))
            plt.show()
    else:
            unique_col_vals = funcs.unique_vals_column(df, col, normalize = False) 
            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 12)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            sns.barplot(x = 'count', y = col , data = unique_col_vals)
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.title('Count of Unique Values in {}'.format(col))
            plt.show()

def count_plot(df, col, **hue):
    """Plot value counts in a column.

    Value counts are calculated for a single column and plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to check for unique values.
    **hue: dict
        Keyword arguments.
    

    Returns
    -------
    seaborn.countplot
    """
    var = hue.get('var', None)
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols and not var:
            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 8)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            sns.countplot(y = df[col], order=df[col].value_counts(ascending=False).index, ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.title('Count of {}'.format(col))
            plt.savefig('images/{}.png'.format(col))
            plt.show()
        elif col in df_cols and var:
            if var in df_cols:
                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(15, 8)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                sns.countplot(y = df[col], order=df[col].value_counts(ascending=False).index, hue = df[var],  ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('{} vs {}'.format(col.title(), var.title()))
                plt.savefig('images/{}v{}.png'.format(col, var))
                plt.show()
            else:
                raise utils.InvalidColumn(var)
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)


def bar_plot(df, col_x, col_y):
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col_x in df_cols:
            if col_y in df_cols:
                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(15, 12)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                bar = sns.barplot(x = col_x, y = col_y, data = df, ci = None, ax=ax)
                # for i in range(len(df)):
                #     bar.text(i, df[col_y][i] + 0.25, str(round(df[col_y][i], 2)),
                #     fontdict= dict(color = 'blue', fontsize = 10, horizontalalignment = 'center'))
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('Rank of {} by {}'.format(col_x, col_y))
                plt.savefig('images/{}_{}.png'.format(col_x, col_y))
                plt.show()
            else:
                raise utils.InvalidColumn(col_y)
        else:
            raise utils.InvalidColumn(col_x)
    else:
        raise utils.InvalidDataFrame(df)


def pie_plot(df, col):
    """Plot pie plot of values in a column.

    Percentage of values counts are calculated for a single column and plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to plot.


    Returns
    -------
    pandas.DataFrame.plot.pie
    """
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols:
            pie_data = df[col].value_counts().reset_index()
            pie_data.set_index('index', inplace=True)
            if len(pie_data) > 5:
                print("{} contains more than 5 values.".format(col))
                print("Visualization best practices recommends using a barplot for variables with more than 5 unique values.")
            elif len(pie_data) <= 5:
                pie_data.plot.pie(y=col, autopct='%0.1f%%', figsize=(10, 10))
                plt.savefig('images/{}.png'.format(col))
                plt.title('{} Composition'.format(col.title()))
                plt.show()
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)

def hist_distribution(df, col, bins = 30, kde = False):
    """Plot distribution of values in a column.

    Histogram with kde(kernel density estimate) of values in a numerical column are plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to plot.
    bins : integer, optional
        Number of bins in distribution.
        (Default value = 30)
    kde : boolean, optional
        If true, a kde(kernel density estimate) plot is included in histogram.
        (Default value = False)

    Returns
    -------
    seaborn.histplot
    """
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols:
            if df[col].dtypes == 'int64' or df[col].dtypes == 'int32' or df[col].dtypes == 'float64':
                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(15, 8)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                sns.histplot(df[col], bins=bins, kde=kde, ax=ax, color="orange")
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('Distribution of {}'.format(col.title()))
                plt.savefig('images/{}_distribution.png'.format(col))
                plt.show()
            else:
                raise utils.InvalidDataType(col)
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)
    

def box_plot(df, col, y = None):
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols:
            if not y:
                if df[col].dtypes == 'int64' or df[col].dtypes == 'int32' or df[col].dtypes == 'float64':
                    fig, ax = plt.subplots()
                    # the size of A4 paper lanscape
                    fig.set_size_inches(15, 8)
                    sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                    sns.boxplot(x = df[col], color = 'green')
                    plt.setp(ax.get_xticklabels(), rotation=0)
                    plt.title('Box Distribution of {}'.format(col.title()))
                    plt.savefig('images/{}_distribution.png'.format(col))
                    plt.show()
                else:
                    raise utils.InvalidDataType(col)
            if y:
                if y in df_cols:
                    if df[y].dtypes == 'object':
                        fig, ax = plt.subplots()
                        # the size of A4 paper lanscape
                        fig.set_size_inches(15, 8)
                        sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                        sns.boxplot(x = df[col], y = df[y])
                        plt.setp(ax.get_xticklabels(), rotation=0)
                        plt.title('{} Distribution by {}'.format(col.title(), y.title()))
                        plt.savefig('images/{}_{}_distribution.png'.format(col,y))
                        plt.show()
                    else:
                        raise utils.InvalidDataType(y)
                else:
                    raise utils.InvalidColumn(y)
        else:
            raise utils.InvalidColumn(col)


def line_plot(df, x, y, hue = None):
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if x in df_cols:
            if y in df_cols:
                if hue:
                    if hue in df_cols:
                        fig, ax = plt.subplots()
                        fig.set_size_inches(15, 8)
                        sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                        sns.lineplot(x = x, y = y, data = df, hue = hue, ci = False)
                        plt.setp(ax.get_xticklabels(), rotation = 90)
                        plt.title('Time Series of Price of BMW used car')
                        plt.savefig('images/tmseriesn.png')
                        plt.show()
                    else:
                        raise utils.InvalidColumn(hue)
                elif not hue:
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 8)
                    sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                    sns.lineplot(x = x, y = y, data = df, ci = False)
                    plt.setp(ax.get_xticklabels(), rotation = 90)
                    plt.title('Time Series of Price of BMW used car')
                    plt.savefig('images/tmseriesn.png')
                    plt.show()
            else:
                raise utils.InvalidColumn(y)
        else:
            raise utils.InvalidColumn(x)
    else:
        raise utils.InvalidDataFrame(df)

def lineplot_by_unique_val(df, col, col_val, hue = None):
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols:
            uniques = funcs.unique_vals_column(df, col, normalize = False)
            if col_val in uniques[col].tolist():
                df_col_val = df.loc[df[col] == col_val]
                if hue:
                    if hue in df_cols:
                        fig, ax = plt.subplots()
                        fig.set_size_inches(15, 8)
                        sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                        sns.lineplot(x = 'year', y = 'price', data = df_col_val, hue = hue, ci = False)
                        plt.setp(ax.get_xticklabels(), rotation = 90)
                        plt.title('Time Series of Price of {} {} {}'.format(col_val.title(), col.title(), hue.title()))
                        plt.savefig('images/{}_{}_tmseries.png'.format(col, col_val))
                        plt.show()
                    else:
                        raise utils.InvalidColumn(hue)
                if not hue:
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 8)
                    sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                    sns.lineplot(x = 'year', y = 'price', data = df_col_val, ci = False)
                    plt.setp(ax.get_xticklabels(), rotation = 90)
                    plt.title('Time Series of Price of {} {}'.format(col_val.title(), col.title()))
                    plt.savefig('images/{}_{}_tmseries.png')
                    plt.show()
            else:
                raise utils.InvalidColumn(col_val)
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidColumn(df)


def corr_heatmap(df, **cols_to_drop):
    """Plot Correlation heatmap.

    Parameters
    ----------
    df : DataFrame
        DataFrame with numerical values to make Correlation Heatmap        

    Returns
    -------
    seaborn.heatmap    
    """
    cols_drop = cols_to_drop.get('cols_drop', None)
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df_cols = df.columns.tolist()
        if not cols_drop:
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 8)
            mask = np.triu(np.ones_like(df.corr(), dtype = bool))
            heatmap = sns.heatmap(df.corr(), mask = mask, vmin = -1, vmax = 1, annot  =True, cmap = 'GnBu')
            heatmap.set_title("Correlation Heatmap of BMW Sales", fontdict = {'fontsize': 16}, pad = 15)
            plt.setp(ax.get_xticklabels(), rotation = 90)
            plt.setp(ax.get_yticklabels(), rotation = 0)
            # plt.savefig("images/dfcorr.png")
            plt.show()
        if cols_drop:
            if isinstance(cols_drop, list):
                if len(cols_drop) == 1:
                    if cols_drop[0] in df_cols:
                        df.drop(cols_drop[0], inplace=True, axis=1)
                        fig, ax = plt.subplots()
                        fig.set_size_inches(15, 8)
                        mask = np.triu(np.ones_like(df.corr(), dtype = bool))
                        heatmap = sns.heatmap(df.corr(), mask = mask, vmin = -1, vmax = 1, annot  =True, cmap = 'Spectral')
                        heatmap.set_title("Correlation Heatmap of BMW Sales", fontdict = {'fontsize': 16}, pad = 15)
                        plt.setp(ax.get_xticklabels(), rotation = 90)
                        plt.setp(ax.get_yticklabels(), rotation = 0)
                        # plt.savefig("images/dfcorr.png")
                        plt.show()
                    else:
                        raise utils.InvalidColumn(cols_drop[0])

                elif len(cols_drop) > 1:
                    col_mems = all(col in df_cols for col in cols_drop)
                    if col_mems:
                        df.drop(cols_drop, inplace=True, axis=1)
                        fig, ax = plt.subplots()
                        fig.set_size_inches(15, 8)
                        mask = np.triu(np.ones_like(df.corr(), dtype = bool))
                        heatmap = sns.heatmap(df.corr(), mask = mask, vmin = -1, vmax = 1, annot  =True, cmap = 'PuOr')
                        heatmap.set_title("Correlation Heatmap of BMW Sales", fontdict = {'fontsize': 16}, pad = 15)
                        plt.setp(ax.get_xticklabels(), rotation = 90)
                        plt.setp(ax.get_yticklabels(), rotation = 0)
                        # plt.savefig("images/dfcorr.png")
                        plt.show()
                    else:
                        non_cols = []
                        for col in cols_drop:
                            if col not in df_cols:
                                raise utils.InvalidColumn(col)
            else:
                raise utils.InvalidDataStructure(cols_drop)
    else:
        raise utils.InvalidDataFrame(df)
