import operator
import os
import numpy as np
import pandas as pd
from os.path import join
import json

#单个任务及其属性
class Task:
    def __init__(self, id):
        self.id = id    #任务id
        self.processor_id = None  #任务分配到的处理器的标识符
        self.rank = None  #任务优先级
        self.comp_cost = []     #任务在不同处理器上的计算成本列表
        self.avg_comp = None    #任务在所有处理器上的平均计算成本
        self.duration = {'start':None, 'end':None}  #任务执行时间（包括开始和结束时间）

#处理器及其状态
class Processor:
    def __init__(self, id):
        self.id = id #处理器的唯一标识符
        self.task_list = [] #该处理器上分配的任务列表

class HEFT: #input_list 用于直接传入 DAG 的相关信息，file 用于传入 DAG 的 DOT 文件路径
    #__init__ 类的构造函数、初始化对象属性DAG 的任务数、处理器数、计算成本矩阵和邻接矩阵
    def __init__(self,num_tasks=10,num_processors=2,data_dir=r"D:\XINY\WORK\电网分核运算\HEFT_xy"):
        self.num_tasks = num_tasks
        self.num_processors = num_processors
        self.comp_cost, self.adj_matrix, self.proc_comm = self.__read_all_matrices(data_dir)

        entry_id = self.__find_ionode() # 如果不止一个头节点，那么会更新节点数量、adj_matrix、comp_cost
        print("计算成本矩阵形状:", len(self.comp_cost), "x", len(self.comp_cost[0]))
        print("邻接矩阵形状:", len(self.adj_matrix), "x", len(self.adj_matrix[0]))
        print("处理器通信矩阵形状:", len(self.proc_comm), "x", len(self.proc_comm[0]))

        # 初始化任务列表和处理器列表
        self.tasks = [Task(i) for i in range(self.num_tasks)]
        self.processors = [Processor(i) for i in range(self.num_processors)]

        # 计算每个任务的平均计算时间成本  对第一维进行遍历，然后求第一维的元素和 所以comp_cost是任务数是行、处理器数量是列
        for i in range(self.num_tasks):
            self.tasks[i].comp_cost = self.comp_cost[i]
            self.tasks[i].avg_comp = sum(self.comp_cost[i]) / self.num_processors #平均计算成本

        self.__computeRanks(self.tasks[entry_id])   # 通过__find_ionode()找到的唯一入口节点
        self.tasks.sort(key=lambda task: (
            0 if task.id == entry_id else 1,  # 入口节点排最前
            -task.rank  # 其他节点按RankU降序（取负号）
        ))
        self._save_rank_to_json(self.tasks,data_dir)

        # 分配处理器并计算makespan
        self.__allotProcessor()
        self.makespan = max([t.duration['end'] for t in self.tasks])    # 遍历所有任务的最后完成时间就是DAG的makespan

        self.__export_schedule_to_json(data_dir)


    #  需要定义这个方法通过邻接矩阵找到DAG的头尾节点，存在多个头尾节点时，则需要创建虚拟头尾节点,其目的是在计算ranku时，有唯一的递归路径
    def __find_ionode(self):
        adj_matrix = np.array(self.adj_matrix)  # 文件读取的只是二维数组 但是后续要使用np方法则是需要将其转化为np数组的形式

        #找出头节点和尾节点的0为列，1为行  某行都是-1那么就是尾节点（无任何节点依赖于他所以尾）， 某列为都为-1则就是头节点（不依赖任何节点则为头）
        # ends = np.nonzero(np.all(self.adj_matrix == -1, axis=1))[0]  # exit nodes 返回的是一个NumPy 数组
        # np.all(adj_matrix == -1, axis=0) 判断每一列的元素是否是-1 是就返回true 不是就false,然后nonzero返回true的索引（元组形式） [0] 的作用是从元组中提取唯一的一维数组，将结果从元组格式转换为一维数组格式。
        starts = np.nonzero(np.all(adj_matrix == -1, axis=0))[0]  # entry nodes

        # 情况判断：根据头尾节点数量决定是否需要构建虚拟节点
        need_dummy_input = len(starts) > 1  # 多个头节点需要虚拟入口

        # 构建虚拟节点逻辑，但凡有一个的len>1就进if条件，
        if need_dummy_input:
            # 获取当前节点数量（用于新节点编号）
            current_num_tasks = self.num_tasks

        # 扩展邻接矩阵，添加新行和新列
            new_size = current_num_tasks + 1
            new_adj_matrix = np.full((new_size, new_size), -1, dtype=int)
            new_adj_matrix[:current_num_tasks, :current_num_tasks] = self.adj_matrix   # 切片操作，复制原矩阵内容[0,current_num_tasks-1]行/列
            # 虚拟入口节点ID为current_num_tasks 因为task是从0开始的，所以新的节点任务id就是current_num_tasks id直接为current_num_tasks
            dummy_input_id = current_num_tasks
            # 将虚拟入口节点连接到所有的源头节点（边权重为0）
            for start_node_id in starts: # 遍历入口节点列表的每一个入口节点 使用邻接矩阵将其所有头节点链接到虚拟入口节点
                new_adj_matrix[dummy_input_id][start_node_id] = 0

        # 扩展计算成本矩阵，新增虚拟入口节点的成本为0
            n_processors = self.num_processors  # 从类属性获取处理器数量
            new_comp_cost = np.zeros((new_size, n_processors))  # 创建新矩阵
            new_comp_cost[:current_num_tasks, :] = self.comp_cost # 复制原始计算成本矩阵 [0，current_num_tasks-1]行  :表示所有列
            new_comp_cost[dummy_input_id, :] = 0 # 第dummy_input_id行i列 每个元素都是0

            # 更新self实例的计算成本矩阵、邻接矩阵、任务数、头节点列表
            self.comp_cost = new_comp_cost
            self.adj_matrix = new_adj_matrix
            self.num_tasks = new_size # 这样的逻辑无论进不进构建头节点的if都不会影响虚拟尾节点的构建逻辑
            # 将 starts 设置为一个 包含dummy_input_id单个元素的 NumPy 数组
            starts = np.array([dummy_input_id])  # 现在只有一个虚拟头节点
            self.dummy_input_id = dummy_input_id    # 添加标志位 以此判断是否有虚拟输入节点

            # 返回最终的头节点（可能是虚拟节点或原始节点）
        return starts[0] # 返回头节点的ID

# 计算节点的ranku 向上排名秩
    def __computeRanks(self, task):
        curr_rank = 0
        for succ in self.tasks: #遍历所有任务的列表，递归计算秩
            if self.adj_matrix[task.id][succ.id] != -1:
                if succ.rank is None:   #且没有计算这个秩
                    self.__computeRanks(succ)   # 一直递归到最后出度为0的节点 慢慢向前递归计算ranku （递归：从后往前）
                #当前节点的ranku 是后续节点的ranku+节点间的数据通信成本 为什么是max中的，因为可能不止一个后续节点
                curr_rank = max(curr_rank, self.adj_matrix[task.id][succ.id] + succ.rank)   #取所有后继任务中 通信成本 + 后继优先级 的最大值
        task.rank = task.avg_comp + curr_rank   #平均计算成本 + 最大后继任务的ranku

# 用于计算任务 t 在处理器 p 上的最早开始时间（EST）
    def __get_est(self, t, p):
        # 无论分配到哪个处理器上，任务节点的est都是需要所有前置任务完成时间+通信时间  以这个est来确定每个处理器上任务start时间
        est = 0
        for pre in self.tasks:
            if self.adj_matrix[pre.id][t.id] != -1:  #遍历每个任务通过邻接矩阵判断是否为t的前置任务 pre是任务 p是处理器
                # c = self.adj_matrix[pre.id][t.id] if pre.processor_id != p.id else 0 # 如果前驱和t不在同一处理器，则加上通信成本
                # 如果pre节点所在的处理器processor_id和处理器p的id不一样，那么c=处理器通信成本（proc_comm）+ 任务间原本的数据通信量（adj_matrix）

                # 如果是虚拟节点那么 c=0
                # getattr
                if pre.id == getattr(self, 'dummy_input_id', None): # 这样写可以解决虚拟节点是否存在的情况 存在不会报错 不存在也不会
                    c = 0
                else:
                    # 原逻辑：如果前驱和t不在同一处理器，加上处理器间通信成本
                    c = self.proc_comm[pre.processor_id][p.id] + self.adj_matrix[pre.id][
                        t.id] if pre.processor_id != p.id else self.adj_matrix[pre.id][t.id]
                est = max(est, pre.duration['end'] + c)  # 取所有前驱完成时间+通信成本的最大值

        free_times = [] #是一个列表，用于存储处理器 p 的所有空闲时间槽
        if len(p.task_list) == 0:
            # 若 p.task_list 为空，整个时间轴 [0, +∞) 都是空闲的
            free_times.append([0, float('inf')])    # free_time是个二元组
        else:
            # 计算已分配任务之间的空闲时间段
            for i in range(len(p.task_list)):
                if i == 0:
                    if p.task_list[i].duration['start'] != 0:
                    # 任务前空闲时间
                        free_times.append([0, p.task_list[i].duration['start']])
                else:
                    # 任务间空闲时间
                    free_times.append([p.task_list[i-1].duration['end'], p.task_list[i].duration['start']])
                    # 任务后的无限空闲时间
            free_times.append([p.task_list[-1].duration['end'], float('inf')])  # 任务列表额最后一个任务结束到 inf

        #遍历 free_times 列表中的每个空闲时间槽slot 所以slot[0]表示空闲时间槽的开始，slot[1]表示空闲时间槽的结束
        for slot in free_times:     # free_times is already sorted based on avaialbe start times
            if est < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]: # 任务的est早于空闲槽起始时间 且任务的计算成本能在这个空闲槽中完成
                return slot[0]  # 任务将在该时间点开始
            if est >= slot[0] and est + t.comp_cost[p.id] <= slot[1]:
                return est  # 任务将在前驱完成后立即开始

#分配任务到处理器的方法
    def __allotProcessor(self):
        for t in self.tasks:    # 遍历降序排列的ranku列表
            if t == self.tasks[0]:   # the one with highest rank DAG入口节点
                p, w = min(enumerate(t.comp_cost), key=operator.itemgetter(1))  # 选择任务节点计算成本最小的处理器 返回处理器p和对应处理器上的计算成本w
                t.processor_id = p
                t.duration['start'] = 0
                t.duration['end'] = w   # 入口节点的end就是计算成本w
                self.processors[p].task_list.append(t)  # 将任务添加到处理器的任务队列中
            else:   # 非头节点的处理
                aft = float("inf")  # 记录当前找到的最早完成时间，初始化为正无穷大。
                for p in self.processors:   # 遍历处理器 计算任务t在其上的est
                    est = self.__get_est(t, p)  #
                    # print("Task: ", t.id, ", Proc: ", p.id, " -> EST: ", est)
                    eft = est + t.comp_cost[p.id]   # 所以节点任务的eft就是 est+处理器上的comp
                    if eft < aft:   # found better case of processor 比较出最小的eft即最佳调度
                        aft = eft
                        best_p = p.id

                # 得到的是节点任务最佳的处理器、节点在处理器上的开始和结束时间
                t.processor_id = best_p
                t.duration['start'] = aft - t.comp_cost[best_p]
                t.duration['end'] = aft
                self.processors[best_p].task_list.append(t)
                # 确保处理器上的任务按开始时间升序排列，便于后续查找空闲时间槽
                self.processors[best_p].task_list.sort(key = lambda x: x.duration['start'])

# 读取xlsx格式的DAG相关信息：DAG邻接矩阵、不同处理器下的通信成本矩阵、不同处理器的通常成本矩阵
    def __read_all_matrices(self, data_dir):
        # 1. 读取计算成本矩阵 NxM 第一维的数据必须是任务数
        comp_path = join(data_dir, "comp_cost.xlsx")
        comp_df = pd.read_excel(comp_path, header=None)

        # 2. 读取邻接矩阵（任务间通信成本）NxN
        adj_path = join(data_dir, "adj_matrix_3.xlsx")
        adj_df = pd.read_excel(adj_path, header=None)

        # 3. 读取处理器间通信成本矩阵 MxM
        proc_path = join(data_dir, "proc_comm.xlsx")
        proc_df = pd.read_excel(proc_path, header=None)

        return (
            comp_df.values.tolist(),
            adj_df.values.tolist(),
            proc_df.values.tolist()
        )

# 保存每个节点的秩及其ranku降序排序
    def _save_rank_to_json(self, sorted_tasks, data_dir):
        rank_data = {
            "description": "任务优先级计算结果 (RankU)",
            "sorted_by_rank": [
                {
                    "rank": i + 1,
                    "task_id": task.id,
                    "rank_value": float(task.rank)
                }
                for i, task in enumerate(sorted_tasks)
            ],
            "schedule_order": [task.id for task in sorted_tasks]
        }

        json_path = join(data_dir, "Tasks_Ranku_2.json")
        with open(json_path, 'w') as f:
            json.dump(rank_data, f, indent=4)

        print(f"\nRanku任务优先级结果已保存到: {json_path}")

# 导出最终的静态任务调度排序情况
    def __export_schedule_to_json(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "DAG_Result_2.json")  # os.path.join路径整合

        schedule_data = {
            "makespan": self.makespan,
            "processors": []
        }

        for p in self.processors:
            processor_data = {
                "id": p.id,
                "tasks": []
            }
            for t in p.task_list:
                task_data = {
                    "id": t.id,
                    "start": t.duration['start'],
                    "end": t.duration['end']
                }
                processor_data["tasks"].append(task_data)
            schedule_data["processors"].append(processor_data)

        with open(file_path, 'w') as f:
            json.dump(schedule_data, f, indent=4)

        print(f"调度结果已保存至: {file_path}")


if __name__ == "__main__":
    new_sch = HEFT()  # 实例化HEFT类
