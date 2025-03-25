################################################################################
# 本文件用于绘图函数的标准化
################################################################################
# 导入模块
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
################################################################################
# 定义基本变量
rcParams['figure.autolayout'] = True
DPI=300                      # 定义图像清晰度
PIC_FORMAT=('pdf', 'tif')    # 定义存储图片的默认格式
################################################################################
def Save_Pictures(
        name="default",
        formats=PIC_FORMAT,
        dpi = DPI
):
    """
    自动保存图片函数
    :param name: 图片名称
    :param formats: 图片格式
    :param dpi: 图片清晰度
    :return: None
    """
    for fmt in formats:
        plt.savefig(
            name+"."+fmt,
            format=fmt,
            dpi=dpi,
            bbox_inches='tight')
    plt.clf()
################################################################################
class Draw_Embedding:
    """
    绘制二维和三维散点图
    """
    def __init__(
        self,
        path='Figure',
        filename='default_name',
        fontsize=8,
        titlefontsize=12,
        title=None,
        cmap='Spectral',
        dota_size=2,
        show_legend = True,
        lgd=None
    ):
        """
        初始化函数
        :param path: 存储图片的很路径
        :param filename: 存储图片的名称
        :param fontsize: 图片中的字号
        :param titlefontsize: 图片中的标题字号
        :param title: 标题
        :param cmap: 颜色风格
        :param dota_size: 散点图中点的大小
        :param show_legend: 是否绘制图例
        :param lgd: 提供的图例列表
        """
        self.path = path
        self.filename = filename
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.cmap = cmap
        self.size = dota_size
        self.show_legend = show_legend
        self.lgd = lgd
        os.makedirs(self.path, exist_ok=True)

    def Draw_embedding(
            self,
            Embedding,
            Target,
            name=None
    ):
        """
        绘制2D散点图
        :param Embedding: 点的坐标 [N, 2]
        :param Target: 与Embedding对应的标签 [N,]
        :param name: 一个含有文件名元素的列表 ["Basic", "1", "UMAP", "USPS"]
        :return: None
        """
        # 优先使用name作为文件名
        if name is not None:
            topic = "-".join(name)
        else:
            topic = self.filename
        # 确定图例列表
        unique_categories, category_counts = np.unique(Target, return_counts=True)
        if self.lgd is not None:
            labels = self.lgd
        else:
            labels = [f"Class {i}" for i in unique_categories]
        # 绘图
        sc = plt.scatter(Embedding[:, 0], Embedding[:, 1], c=Target.astype(int), s=self.size,cmap=self.cmap)
        # 调整图片
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        if self.show_legend:
            plt.legend(handles=sc.legend_elements(num=len(unique_categories)-1)[0], loc='upper right', labels=labels, ncol=np.ceil(len(unique_categories)/10), fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, topic))

    def Draw_embedding_3D(
            self,
            Embedding,
            Target,
            name=None
    ):
        """
        绘制3D散点图
        :param Embedding: 点的坐标 [N, 3]
        :param Target: 与Embedding对应的标签 [N,]
        :param name: 一个含有文件名元素的列表 ["Basic", "1", "UMAP", "USPS"]
        :return: None
        """
        # 优先使用name作为文件名
        if name is not None:
            topic = "-".join(name)
        else:
            topic = self.filename
        # 确定图例列表
        unique_categories, category_counts = np.unique(Target, return_counts=True)
        if self.lgd is not None:
            labels = self.lgd
        else:
            labels = [f"Class {i}" for i in unique_categories]
        # 绘图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Embedding[:, 0], Embedding[:, 1], Embedding[:, 2], s=self.size, c=Target.astype(int), marker='o')
        # 调整图片
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        if self.show_legend:
            plt.legend(handles=sc.legend_elements(num=len(category_counts))[0], ncol=np.ceil(len(category_counts) / 10), loc='upper right', labels=labels, fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, topic))
################################################################################
class Draw_Pairhist:
    """
    绘制x轴上下对比的直方图
    """
    def __init__(
        self,
        path='Figure',
        filename='default',
        fontsize=8,
        titlefontsize=12,
        bar_num=30
    ):
        """
        初始化函数
        :param path: 存储图片的很路径
        :param filename: 存储图片的名称
        :param fontsize: 图片中的字号
        :param titlefontsize: 图片中的标题字号
        :param bar_num: 柱的数量
        """
        self.path = path
        self.filename = filename
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.bar_num = bar_num
        os.makedirs(self.path, exist_ok=True)

    def format_func(self, value, tick_number):
        """
        格式化纵轴刻度
        :param value: 刻度值
        :return: None
        """
        return "{:,.0e}".format(abs(value))

    def Draw_pairhist(self, dist_train, dist_test):
        """
        绘制直方图
        :param dist_train: x轴上方的距离矩阵
        :param dist_test: x轴下方的距离矩阵
        :return: None
        """
        # 计算值的范围
        self.range_min = np.min(np.array([np.min(dist_train), np.min(dist_test)]))
        self.range_max = np.max(np.array([np.max(dist_train), np.max(dist_test)]))
        # 绘图
        plt.figure()
        hist_up, bins_up = np.histogram(dist_train.ravel(), bins=self.bar_num, range=(self.range_min, self.range_max))
        bin_centers_up = (bins_up[:-1] + bins_up[1:]) / 2
        plt.bar(bin_centers_up, hist_up, width=(bins_up[1] - bins_up[0]), color="#336872", edgecolor="white", label='train-data')
        hist_down, bins_down = np.histogram(dist_test.ravel(), bins=self.bar_num, range=(self.range_min, self.range_max))
        bin_centers_down = (bins_down[:-1] + bins_down[1:]) / 2
        plt.bar(bin_centers_down, -hist_down, width=(bins_down[1] - bins_down[0]), color="#EF7B30", edgecolor="white", label='oos-data')
        # 调整图片
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.format_func))
        plt.legend(fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.xlabel('Distance values', fontsize=self.titlefontsize)
        plt.ylabel('Distance values in that block', fontsize=self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
class CustomLegendHandler:
    """
    调整折线图的图例
    """
    def __init__(
            self,
            color,
            marker,
            linestyle,
            label,
            markeredgecolor,
            markerfacecolor
    ):
        """
        初始化函数
        :param color:
        :param marker:
        :param linestyle:
        :param label:
        :param markeredgecolor:
        :param markerfacecolor:
        """
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.label = label
        self.markeredgecolor = markeredgecolor
        self.markerfacecolor = markerfacecolor

    def create_proxy_artist(self, legend):
        return Line2D(
            [0], [0],
            color=self.color,
            marker=self.marker,
            linestyle=self.linestyle,
            markeredgecolor=self.markeredgecolor,
            markerfacecolor=self.markerfacecolor)

class Draw_Line_Chart:
    """
    折线图绘制类
    """
    def __init__(
        self,
        path='Figure',
        filename='default',
        fontsize=8,
        titlefontsize=12,
        left=None, right=None, column=None,
        left_label=None, right_label=None,
        ylim_left=(0,1), ylim_right=(0,1),
        left_color=None, right_color=None,
        left_marker=None, right_marker=None,
        left_markeredgecolor=None, right_markeredgecolor=None,
        left_markerfacecolor=None, right_markerfacecolor=None,
        xlabel='% size of train set',
        ylabel_left='classification accuracy',
        ylabel_right='clustering score',
        title = None
    ):
        """
        初始化函数
        :param path: 图片存储路径
        :param filename: 图片文件名
        :param fontsize: 图中文字大小
        :param titlefontsize: 图中标题文字大小
        :param left: 左侧纵轴绘制的指标
        :param right: 右侧纵轴绘制的指标
        :param column: 横轴的值列表
        :param left_label: 左轴指标的标签
        :param right_label: 右轴指标的标签
        :param ylim_left: 左轴的限制范围
        :param ylim_right: 右轴的限制范围
        :param left_color: 左轴的线条颜色
        :param right_color: 右轴的线条颜色
        :param left_marker: 左轴的标记的形状
        :param right_marker: 右轴的标记的形状
        :param left_markeredgecolor: 左轴的标记的边颜色
        :param right_markeredgecolor: 右轴的标记的边颜色
        :param left_markerfacecolor: 左轴的标记的面颜色
        :param right_markerfacecolor: 右轴的标记的面颜色
        :param xlabel: 横轴的标签
        :param ylabel_left: 左轴的标签
        :param ylabel_right: 右轴的标签
        :param title: 标题
        """
        self.path = path
        self.filename = filename
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.left = np.array(left)
        self.right = np.array(right)
        self.column = np.array(range(1, self.left.shape[1] + 1)) if column is None else column
        self.left_num = 0 if left is None else len(left)
        self.right_num = 0 if right is None else len(right)
        self.left_label = ["left-"+str(i+1) for i in range(self.left_num)] if left_label is None else left_label
        self.right_label = ["right-"+str(i+1) for i in range(self.right_num)] if right_label is None else right_label
        self.left_color = np.random.random((self.left_num, 3)) if left_color is None else left_color
        self.right_color = np.random.random((self.right_num, 3)) if right_color is None else right_color
        self.left_markeredgecolor = self.left_color if left_markeredgecolor is None else left_markeredgecolor
        self.right_markeredgecolor = self.right_color if right_markeredgecolor is None else right_markeredgecolor
        self.left_markerfacecolor = self.left_color if left_markerfacecolor is None else left_markerfacecolor
        self.right_markerfacecolor = self.right_color if right_markerfacecolor is None else right_markerfacecolor
        self.left_marker = ["o"]*self.left_num if left_marker is None else left_marker
        self.right_marker = ["o"]*self.right_num if right_marker is None else right_marker
        self.xlabel = xlabel
        self.ylabel_left = ylabel_left
        self.ylabel_right = ylabel_right
        self.ylim_left = ylim_left
        self.ylim_right = ylim_right
        self.title = title
        os.makedirs(self.path, exist_ok=True)

    def Draw_double_line(self):
        """
        绘制双轴折线图
        :return:
        """
        # 定义坐标轴
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # 绘制左轴
        for i in range(self.left_num):
            ax1.plot(self.column, self.left[i], color=self.left_color[i],linestyle='-', label=self.left_label[i])
            ax1.scatter(self.column, self.left[i], facecolor=self.left_markerfacecolor[i], edgecolor=self.left_markeredgecolor[i], marker=self.left_marker[i])
        if self.ylim_left is not None:
            ax1.set_ylim(self.ylim_left[0], self.ylim_left[1])
        ax1.tick_params(axis='y')
        ax1.set_xlabel(self.xlabel, fontsize = self.titlefontsize)
        ax1.set_ylabel(self.ylabel_left, fontsize = self.titlefontsize)
        # 绘制右轴
        for i in range(self.right_num):
            ax2.plot(self.column, self.right[i], color=self.right_color[i],linestyle='--', label=self.right_label[i])
            ax2.scatter(self.column, self.right[i], facecolor=self.right_markerfacecolor[i], edgecolor=self.right_markeredgecolor[i], marker=self.right_marker[i])
        if self.ylim_right is not None:
            ax2.set_ylim(self.ylim_right[0], self.ylim_right[1])
        ax2.tick_params(axis='y')
        ax2.set_ylabel(self.ylabel_right, fontsize = self.titlefontsize)
        # 定义图例
        custom_legend_handles = []
        for color, marker, linestyle, label, markeredgecolor, markerfacecolor in zip(self.left_color, self.left_marker, ['-'] * self.left_num, self.left_label, self.left_markeredgecolor, self.left_markerfacecolor):
            custom_legend_handles.append(CustomLegendHandler(color, marker, linestyle, label, markeredgecolor, markerfacecolor))
        for color, marker, linestyle, label, markeredgecolor, markerfacecolor in zip(self.right_color, self.right_marker, ['--'] * self.right_num, self.right_label, self.right_markeredgecolor, self.right_markerfacecolor):
            custom_legend_handles.append(CustomLegendHandler(color, marker, linestyle, label, markeredgecolor, markerfacecolor))
        proxy_artists = [handler.create_proxy_artist(ax2) for handler in custom_legend_handles]
        ax2.legend(proxy_artists, self.left_label + self.right_label, loc="best", fontsize = self.fontsize)
        # 定义刻度
        plt.xticks(fontsize = self.fontsize)
        plt.yticks(fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))

    def Draw_simple_line(self):
        """
        定义单轴折线图
        :return:
        """
        fig, ax = plt.subplots()
        # 绘图
        for i in range(self.left_num):
            plt.plot(np.array(self.column), np.array(self.left[i]), color=self.left_color[i], linestyle='-', label=self.left_label[i])
            plt.scatter(np.array(self.column), np.array(self.left[i]), facecolor=self.left_markerfacecolor[i], edgecolor=self.left_markeredgecolor[i], marker=self.left_marker[i], label=self.left_label[i])
        # 定义图例
        custom_legend_handles = []
        for color, marker, linestyle, label, markeredgecolor, markerfacecolor in zip(self.left_color, self.left_marker, ['-'] * self.left_num, self.left_label, self.left_markeredgecolor, self.left_markerfacecolor):
            custom_legend_handles.append(CustomLegendHandler(color, marker, linestyle, label, markeredgecolor, markerfacecolor))
        proxy_artists = [handler.create_proxy_artist(ax) for handler in custom_legend_handles]
        ax.legend(proxy_artists, self.left_label + self.right_label, loc="best", fontsize=self.fontsize)
        # 调整图形
        if self.ylim_left is not None:
            plt.ylim(self.ylim_left[0], self.ylim_left[1])
        plt.tick_params(axis='y')
        plt.xlabel(self.xlabel, fontsize = self.titlefontsize)
        plt.ylabel(self.ylabel_left, fontsize = self.titlefontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
class Confusion_Matrix:
    """
    混淆矩阵绘制
    """
    def __init__(
        self,
        path='Figure',
        filename='default',
        fontsize=8,
        titlefontsize=12,
        title=None,
        lgd = None
    ):
        """
        初始化函数
        :param path: 图片存储路径
        :param filename: 图片文件名
        :param fontsize: 图中文字大小
        :param titlefontsize: 图中标题文字大小
        :param title: 图片标题
        :param lgd: 图例列表
        """
        self.path = path
        self.filename = filename
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.lgd = lgd
        os.makedirs(self.path, exist_ok=True)

    def Drawing(self, true_label, predict_label):
        """
        绘制混淆矩阵主函数
        :param true_label: 真实标签
        :param predict_label: 预测标签
        :return:
        """
        self.true_label = true_label
        self.predict_label = predict_label
        self.Calculate()
        cmap = ListedColormap(['#3564A7', '#FFFFFF', '#759FCC'])
        # 定义刻度
        unique_categories, category_counts = np.unique(true_label, return_counts=True)
        if self.lgd is not None:
            xticklabels = self.lgd
            yticklabels = self.lgd
        else:
            xticklabels = yticklabels = [f"Class {i}" for i in unique_categories]
        # 绘制混淆矩阵
        plt.figure()
        sns.heatmap(
            self.color_matrix,
            annot=False, fmt='d',
            cbar=False, cmap=cmap,
            linecolor='#7F7F7F',
            linewidths=0.5
        )
        # 调整图片
        plt.xticks(ticks=np.array(range(len(xticklabels)))+0.5, labels=xticklabels, fontsize = self.fontsize)
        plt.yticks(ticks=np.array(range(len(yticklabels)))+0.5, labels=yticklabels, fontsize = self.fontsize, rotation=0)
        plt.xlabel("Predict Class", fontsize = self.titlefontsize)
        plt.ylabel("True Class", fontsize = self.titlefontsize)
        # 打印文字
        text_font_size = 12 if self.fontsize > 12 else self.fontsize
        plt.text(self.num_classes + 0.5, self.num_classes + 0.5, f"{self.total_ratio * 100:.2f}%", fontsize=text_font_size, horizontalalignment='center', verticalalignment='center', color='white')
        for i in range(len(self.cm)):
            for j in range(len(self.cm)):
                text = f"{self.cm[i, j]}\n({self.cell_proportion[i, j] * 100:.2f}%)"
                plt.text(j + 0.5, i + 0.5, text, fontsize=text_font_size, horizontalalignment='center', verticalalignment='center', color='white' if i == j else 'black')
        for i in range(self.num_classes):
            plt.text(self.num_classes + 0.5, i + 0.5, f"{self.row_ratios[i] * 100:.2f}%", fontsize=text_font_size, horizontalalignment='center', verticalalignment='center', color='white')
            plt.text(i + 0.5, self.num_classes + 0.5, f"{self.column_ratios[i] * 100:.2f}%", fontsize=text_font_size, horizontalalignment='center', verticalalignment='center', color='white')
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))

    def Calculate(self):
        """
        根据真实标签与预测标签计算混淆矩阵
        :return: 混淆矩阵
        """
        self.cm = confusion_matrix(self.true_label, self.predict_label)
        self.num_classes = len(self.cm)
        self.cell_proportion = self.cm / self.cm.sum()
        self.cm_with_ratios = np.zeros((self.num_classes + 1, self.num_classes + 1))
        self.cm_with_ratios[:self.num_classes, :self.num_classes] = self.cm
        self.column_ratios = np.diag(self.cm_with_ratios)[:-1] / self.cm.sum(axis=0)
        self.row_ratios = np.diag(self.cm_with_ratios)[:-1] / self.cm.sum(axis=1)
        self.total_ratio = np.sum(np.diag(self.cm_with_ratios)) / self.cm.sum()
        self.color_matrix = np.eye(len(self.cm_with_ratios))
        self.color_matrix[-1, :] = -1
        self.color_matrix[:, -1] = -1
        self.color_matrix += 1
################################################################################
class Annotated_Heatmaps:
    """
    热图绘制
    """
    def __init__(
        self,
        path='Figure',
        filename='default',
        fontsize=8,
        titlefontsize=12,
        title=None,
        xlabel = None,
        ylabel = None,
        xticklabels=None,
        yticklabels=None,
        cmap=('#FFFFFF', '#3564A7')
    ):
        """
        初始化函数
        :param path: 存储图片的路径
        :param filename: 图片的文件名
        :param fontsize: 图中字体的大小
        :param titlefontsize: 图中标题字体的大小
        :param title: 图中标题
        :param xlabel: 横轴标签
        :param ylabel: 纵轴标签
        :param xticklabels: 横轴刻度
        :param yticklabels: 纵轴刻度
        :param cmap: 颜色映射
        """
        self.path = path
        self.filename = filename
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticklabels = xticklabels
        self.yticklabels = yticklabels
        self.cmap = cmap
        os.makedirs(self.path, exist_ok=True)

    def Drawing(self, harvest):
        """
        绘制热图主函数
        :param harvest: 待绘制的矩阵
        :return:
        """
        # 确定颜色映射
        if isinstance(self.cmap, tuple) and len(self.cmap) == 2:
            cmap = create_colormap(bottom=self.cmap[0], top=self.cmap[1])
        else:
            cmap = self.cmap
        # 确定刻度
        if self.xticklabels is None:
            self.xticklabels = [str(i) for i in range(harvest.shape[1])]
            self.yticklabels = [str(i) for i in range(harvest.shape[0])]
        # 绘图
        plt.figure()
        sns.heatmap(
            harvest,
            annot=False,
            fmt='d',
            cbar=False,
            cmap=cmap,
            linecolor='#7F7F7F',
            linewidths=0.5
        )
        # 调整图形
        plt.xticks(ticks=np.array(range(len(self.xticklabels)))+0.5, labels=self.xticklabels, fontsize = self.fontsize)
        plt.yticks(ticks=np.array(range(len(self.yticklabels)))+0.5, labels=self.yticklabels, fontsize = self.fontsize, rotation=0)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel, fontsize = self.titlefontsize)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel, fontsize = self.titlefontsize)
        # 打印文字
        for i in range(harvest.shape[0]):
            for j in range(harvest.shape[1]):
                text = f"({harvest[i, j] * 100:.2f}%)"
                plt.text(
                    j + 0.5, i + 0.5, text,
                    fontsize=self.fontsize,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black'
                )
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
class Color_Mapping:
    """
    颜色图绘制
    """
    def __init__(
        self,
        path='Figure',
        filename='default',
        fontsize=8,
        titlefontsize=12,
        title = None,
        lgd=None
    ):
        """
        初始化函数
        :param path: 存储图片的路径
        :param filename: 图片的文件名
        :param fontsize: 图中的字体大小
        :param titlefontsize: 图中标题字体的大小
        :param title: 标题
        :param lgd: 图例列表
        """
        self.path = path
        self.filename = filename
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.lgd = lgd
        os.makedirs(self.path, exist_ok=True)

    def Calculate(self):
        """
        计算结果矩阵
        :return:
        """
        self.unique_categories, self.category_counts = np.unique(self.true_label, return_counts=True)
        self.matrix = np.zeros((len(self.unique_categories), self.category_counts[0]))
        for c, category in enumerate(self.unique_categories):
            indices = np.where(self.true_label == category)[0]
            label = self.predict_label[indices]
            self.matrix[c, :] = label

    def Mapping(self, true_label, predict_label):
        """
        绘制色图的主函数
        :param true_label: 真实标签
        :param predict_label: 预测标签
        :return:
        """
        self.true_label = true_label
        self.predict_label = predict_label
        self.Calculate()
        plt.figure()
        plt.imshow(self.matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        if self.lgd is not None:
            yticklabels = self.lgd
        else:
            yticklabels = [f"Class {i}" for i in self.unique_categories]
        plt.ylabel("True Class", fontsize = self.titlefontsize)
        plt.yticks(ticks=list(range(len(yticklabels))), labels=yticklabels, fontsize = self.fontsize)
        plt.xticks(fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
def create_colormap(bottom="blue", top="yellow"):
    """
    创建自定义colormap
    :param bottom: 底端颜色
    :param top: 顶端颜色
    :return: colormap
    """
    colors = [bottom, top]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    return cmap
################################################################################
class Error_Drawing:
    """
    误差图绘制
    """
    def __init__(
        self,
        path='Figure',
        filename='default',
        fontsize=8,
        titlefontsize=12,
        title=None,
        xlabel="x",
        ylabel="y",
        formats = None
    ):
        """
        初始化函数
        :param path: 图片存储的路径
        :param filename: 图片的文件名
        :param fontsize: 图中文字的大小
        :param titlefontsize: 图中标题文字的大小
        :param title: 图片标题
        :param xlabel: 横轴标签
        :param ylabel: 纵轴标签
        :param formats: 图片存储格式
        """
        self.filename = filename
        self.path = path
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.formats = formats if formats else ["tif", "pdf"]
        os.makedirs(self.path, exist_ok=True)

    def drawing_banding(
            self,
            x_value,
            mean_value,
            std_value,
            colors=None,
            markers=None,
            labels=None
    ):
        """
        绘制带有误差带的折线图
        :param x_value: 横轴的值
        :param mean_value: 指标的均值
        :param std_value: 误差、标准差
        :param colors: 颜色列表
        :param markers: 标记列表
        :param labels: 图例列表
        :return:
        """
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        markers = ["o"] * mean_value.shape[1] if markers is None else markers
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        plt.figure()
        for i in range(mean_value.shape[0]):
            plt.plot(
                x_value, mean_value[i], label="Mean of " + labels[i],
                color=colors[i], marker=markers[i])
            plt.fill_between(
                x_value, mean_value[i] - std_value[i],
                mean_value[i] + std_value[i], color=colors[i],
                alpha=0.2, label="Std of " + labels[i])
        plt.xlabel(self.xlabel, fontsize=self.titlefontsize)
        plt.ylabel(self.ylabel, fontsize=self.titlefontsize)
        plt.ylim(0, 1.05)
        plt.title(self.title, fontsize=self.titlefontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.legend(
            loc='lower right',
            fontsize=12
        )
        plt.grid(True)
        Save_Pictures(os.path.join(self.path, self.filename), formats=self.formats)

    def drawing_line_error(
            self,
            x_value,
            mean_value,
            std_value,
            colors=None,
            labels=None
    ):
        """
        绘制带有误差棒的折线图
        :param x_value: 横轴的值
        :param mean_value: 指标的均值
        :param std_value: 误差、标准差
        :param colors: 颜色列表
        :param labels: 图例列表
        :return:
        """
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        plt.figure()
        for i in range(mean_value.shape[0]):
            plt.errorbar(
                x_value, mean_value[i], yerr=std_value[i],
                color=colors[i], label=labels[i])
        plt.xlabel(self.xlabel, fontsize=self.titlefontsize)
        plt.ylabel(self.ylabel, fontsize=self.titlefontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.ylim(0, 1.05)
        plt.title(self.title, fontsize=self.titlefontsize)
        plt.legend(fontsize=self.fontsize)
        plt.grid(True)
        Save_Pictures(os.path.join(self.path, self.filename), formats=self.formats)

    def drawing_bar_error(
            self,
            x_value,
            mean_value,
            std_value,
            colors=None,
            labels=None,
            ylim_flag=True
    ):
        """
        绘制带有误差棒的柱状图
        :param x_value: 横轴的值
        :param mean_value: 指标的均值
        :param std_value: 误差、标准差
        :param colors: 颜色列表
        :param labels: 图例列表
        :param ylim_flag: 是否限制y轴的范围
        :return:
        """
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        x = np.arange(mean_value.shape[1])
        width = 0.80 / mean_value.shape[0]
        fig, ax = plt.subplots(layout='constrained')
        for i in range(mean_value.shape[0]):
            ax.bar(x + width * i, mean_value[i], width, yerr=std_value[i], color=colors[i], label=labels[i])
        ax.set_xlabel(self.xlabel, fontsize=self.titlefontsize)
        ax.set_ylabel(self.ylabel, fontsize=self.titlefontsize)
        if ylim_flag:
            ax.set_ylim(0, 1.05)
        ax.set_title(self.title, fontsize=self.titlefontsize)
        ax.set_xticks(x+0.4-width/2, x_value, rotation=45)
        plt.tick_params(axis='both', labelsize=self.fontsize)
        plt.legend(
            loc='upper center',
            ncol=2,
            fontsize=self.fontsize
        )
        Save_Pictures(os.path.join(self.path, self.filename), formats=self.formats)
        plt.close()

    def drawing_barh_error(
            self,
            x_value,
            mean_value,
            std_value,
            colors=None,
            labels=None,
            xlim_flag=True
    ):
        """
        绘制带有误差棒的横向柱状图
        :param x_value: 纵轴的值
        :param mean_value: 指标的均值
        :param std_value: 误差、标准差
        :param colors: 颜色列表
        :param labels: 图例列表
        :param xlim_flag: 是否限制x轴的范围
        :return:
        """
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        x = np.arange(mean_value.shape[1])
        width = 0.80 / mean_value.shape[0]
        fig, ax = plt.subplots(layout='constrained')
        for i in range(mean_value.shape[0]):
            ax.barh(x + width * i, mean_value[i], width, xerr=std_value[i], color=colors[i], label=labels[i])
        ax.set_xlabel(self.ylabel, fontsize=self.titlefontsize)
        ax.set_ylabel(self.xlabel, fontsize=self.titlefontsize)
        ax.set_title(self.title, fontsize=self.titlefontsize)
        ax.set_yticks(x+0.4-width/2, x_value)
        plt.tick_params(axis='both', labelsize=self.fontsize)
        if xlim_flag:
            ax.set_xlim(0, 1.05)
        plt.legend(fontsize=self.fontsize)
        Save_Pictures(os.path.join(self.path, self.filename), formats=self.formats)
        plt.close()
################################################################################
class Visual_Pixes:
    """
    可视化数据集
    """
    def __init__(
        self,
        path="Figure",
        filename="Visualization-Datssets",
        fontsize=8,
        titlefontsize=12,
        pic_height=32,
        pic_weight=32,
        total_weight=10,
        lgd = None,
        title= None
    ):
        """
        初始化函数
        :param path: 存储图片的路径
        :param filename: 存储图片的文件名
        :param fontsize: 图中文字的大小
        :param titlefontsize: 图中标题文字的大小
        :param pic_height: 单张图片的高度
        :param pic_weight: 单张图片的宽度
        :param total_weight: 每类样本的个数
        :param lgd: 刻度列表
        :param title: 图片标题
        """
        self.path = path
        self.filename = filename
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.pic_height = pic_height
        self.pic_weight = pic_weight
        self.total_weight = total_weight
        self.lgd = lgd
        self.title= title
        os.makedirs(path, exist_ok=True)

    def Calculate_Pixes(self, data, target):
        """
        整理像素矩阵
        :param data: 数据
        :param target: 标签
        :return: 像素矩阵
        """
        data = data.reshape(data.shape[0], self.pic_height, self.pic_weight)
        self.unique_categories, self.category_counts = np.unique(target, return_counts=True)
        self.n = len(self.unique_categories)
        vline = np.ones((self.pic_height, 1))
        temph = np.ones((1, self.total_weight*(self.pic_weight+1)+1))
        hline = np.ones((1, self.total_weight*(self.pic_weight+1)+1))
        for i in self.unique_categories:
            data_i = data[target == i]
            tempv = np.zeros((self.pic_height, 1))
            for j in range(self.total_weight):
                tempv = np.concatenate((tempv, data_i[j]), axis=1)
                tempv = np.concatenate((tempv, vline), axis=1)
            temph = np.concatenate((temph, tempv), axis=0)
            temph = np.concatenate((temph, hline), axis=0)
        return temph

    def Drawing(self, data, target):
        """
        绘制主函数
        :param data: 数据
        :param target: 标签
        :return:
        """
        matrix = self.Calculate_Pixes(data, target)
        plt.imshow(matrix)
        if self.lgd is not None:
            yticklabels = self.lgd
        else:
            yticklabels = [f"Class {i}" for i in self.unique_categories]
        xticklabels = [f"Sample {i}" for i in range(1, self.total_weight+1)]
        plt.xticks(ticks=list(range(1+self.pic_weight//2, (self.pic_weight+1)*self.total_weight+1, self.pic_weight+1)), labels=xticklabels, fontsize=self.fontsize, rotation=30)
        plt.yticks(ticks=list(range(1+self.pic_height//2, (self.pic_height+1)*self.n+1, self.pic_height+1)), labels=yticklabels, fontsize=self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize=self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
