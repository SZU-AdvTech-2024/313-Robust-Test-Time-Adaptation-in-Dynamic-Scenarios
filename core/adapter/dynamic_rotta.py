import torch
from .rotta import RoTTA

class DynamicMemorySizeRoTTA(RoTTA):
    def __init__(self, cfg, model, optimizer):
        # 初始化参数
        super().__init__(cfg, model, optimizer)
        self.initial_memory_size = 64  # 初始的 MEMORY_SIZE
        self.memory_size = self.initial_memory_size  # 当前的 MEMORY_SIZE
        self.previous_accuracies = []  # 用来保存最近几批的准确度
        self.num_recent = 5  # 保持最近 5 个批次的准确度
        self.accuracy_threshold_up = 1.1  # 如果准确度上升超过10%，减少 MEMORY_SIZE
        self.accuracy_threshold_down = 0.95  # 如果准确度下降超过5%，增加 MEMORY_SIZE
        self.min_memory_size = 64  # MEMORY_SIZE 最小值
        self.max_memory_size = 512  # MEMORY_SIZE 最大值

    def forward_and_adapt(self, batch_data, model, optimizer):
        """重写forward_and_adapt方法，加入动态MEMORY_SIZE调整"""
        # 确认 batch_data 是否为 Tensor
        if isinstance(batch_data, torch.Tensor):
            inputs = batch_data
            true_labels = None  # 仅有图像数据，没有标签
        else:
            inputs, true_labels = batch_data  # 如果是其他类型（如元组），则解包

        # 继续执行原始的 RoTTA 适应过程
        ema_out = super().forward_and_adapt(batch_data, model, optimizer)

        if true_labels is not None:
            predicted_labels = torch.argmax(ema_out, dim=1)
            true_labels = true_labels.view(-1)
            predicted_labels = predicted_labels.view(-1)
            current_error_rate = self.calculate_error_rate(predicted_labels, true_labels)
            self.adjust_memory_size(current_error_rate)

        return ema_out

    def calculate_accuracy(self, predicted_labels, true_labels):
        """计算准确度"""
        predicted_labels = torch.argmax(predicted_labels, dim=1)
        accurate = (predicted_labels == true_labels).float()
        accuracy = accurate.mean().item()
        return accuracy

    def adjust_memory_size(self, current_accuracy):
        """动态调整 MEMORY_SIZE"""
        # 保存当前批次的准确率
        self.previous_accuracies.append(current_accuracy)

        # 如果记录的准确度超过 num_recent 个批次，则开始计算平均准确度
        if len(self.previous_accuracies) > self.num_recent:
            self.previous_accuracies.pop(0)

        if len(self.previous_accuracies) == self.num_recent:
            avg_accuracy = sum(self.previous_accuracies) / len(self.previous_accuracies)

            # 判断当前准确度的变化趋势，并据此调整 MEMORY_SIZE
            if current_accuracy < avg_accuracy * self.accuracy_threshold_down:  # 准确度下降超过10%
                self.memory_size = min(self.memory_size * 1.1, self.max_memory_size)  # 增加 MEMORY_SIZE
            elif current_accuracy > avg_accuracy * self.accuracy_threshold_up:  # 准确度上升超过10%
                self.memory_size = max(self.memory_size * 0.9, self.min_memory_size)  # 减少 MEMORY_SIZE

            # 更新内存池大小
            self.update_memory_bank()

    def update_memory_bank(self):
        """更新内存池的大小"""
        if hasattr(self, "mem"):
            self.mem.set_size(self.memory_size)  # 假设有一个 set_size 方法来更新内存池大小

    def get_memory_size(self):
        """获取当前的 MEMORY_SIZE"""
        return self.memory_size
