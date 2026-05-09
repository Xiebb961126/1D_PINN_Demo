import torch
import numpy as np
import json
import torch.nn as nn
import torch.autograd
import torch.optim as optim


class OneDPINNv2(nn.Module):
    def __init__(self, vessel_number: int, finite_number: int, setting_list: list, act_func: str):
        super(OneDPINNv2, self).__init__()
        """
        :param vessel_number: 网络血管的个数
        :param finite_number: 划分的有限元个数
        :param setting_list: 每个血管NN的设置 [input_dim, hidden_dim1, ..., output_dim]
        :param act_func: 激活函数
        """
        self.vessel_number = vessel_number
        self.finite_number = finite_number
        self.act_func = act_func
        self.layer_setting = []
        for i in range(len(setting_list) - 1):
            self.layer_setting.append((setting_list[i], setting_list[i + 1]))
        self.model_list = nn.Sequential()
        self.init_network_layer()
        self.weight_init()

    def normalization(self, x, area):
        """
        :param x: (vessel_num, point_num)
        :param area: (vessel_num, point_num)
        :return:
        """
        number_vessel, _ = area.shape
        area_mean = torch.mean(area)  # (8, 1)
        L = torch.sqrt(area_mean)

        # x normalization
        nor_x = torch.zeros_like(x, device=x.device)
        nor_x[:, :] = x / L
        x_mean = torch.mean(nor_x)
        x_std = torch.std(nor_x)
        nor_x = nor_x - x_mean
        nor_x = nor_x / x_std

        # area normalization
        nor_area = torch.zeros_like(area, device=area.device)
        nor_area[:, :] = area / area_mean  # (8,401,1）
        return nor_x, nor_area

    def init_network_layer(self):
        """
            为整个血管树创建一个fullconv
        :return:
        """
        # 输入的x.shape为[vessel_number, point_num], 所以网络输入[x, area]为[vessel_number, point_num， 2]
        num_setting = len(self.layer_setting)
        for i, (in_dim, out_dim) in enumerate(self.layer_setting):
            in_ch, out_ch = in_dim * self.vessel_number, out_dim * self.vessel_number
            self.model_list.add_module('conv_%d' % i,
                                       nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=self.vessel_number))
            if i < num_setting - 1:
                self.model_list.add_module('act_%d' % i, self.select_activate_function())

    def select_activate_function(self):
        if self.act_func == "tanh":
            return nn.Tanh()
        elif self.act_func == "sigmoid":
            return nn.Sigmoid()
        elif self.act_func == "relu":
            return nn.ReLU()
        else:
            print("error: none defined activate function")

    def weight_init(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        for model in self.model_list:
            model.apply(init)

    def forward(self, x: torch.FloatTensor, area: torch.FloatTensor, measure_v: list,
                measure_p: list, interfaces: list, p_norm, Q_norm):
        """
        :param x: (vessel_num, point_num)
        :param area: (vessel_num, point_num)
        :param measure_v: (meas_num, [vessel_id, point_id, value]) 第几个测量点，加在第几个血管的第几个点上
        :param measure_p: (meas_num, [vessel_id, point_id, value])  第几个测量点，加在第几个血管的第几个点上
        :param interfaces: (interface_num, [father, [son1, ..., son_n]]]) 父亲血管id和儿子血管id
        :return:
        """

        x.requires_grad = True
        nor_x, nor_area = self.normalization(x, area)
        input_tensor = torch.stack((x, area), dim=-1)
        nor_input_tensor = torch.stack((nor_x, nor_area), dim=-1)  # vessel_num, point_num, 2
        nor_input_tensor = nor_input_tensor.transpose(1, 2)  # vessel_num, 2, point_num
        vessel_num, _, point_num = nor_input_tensor.shape
        nor_input_tensor = nor_input_tensor.reshape((-1, point_num))  # vessel_num * 2, point_num
        output_tensor = self.model_list(nor_input_tensor.unsqueeze(0))  # 1, vessel_num * 2, point_num
        output_tensor = output_tensor.squeeze(0).reshape((vessel_num, -1,  point_num))  # vessel_num, 2, point_num
        output_tensor = output_tensor.transpose(1, 2)  # vessel_num, point_num, 2

        output_tensor[:, :, 0] = torch.sigmoid(output_tensor[:, :, 0]) * 2
        output_tensor[:, :, 1] = measure_p[0][-1] * (1 - torch.sigmoid(output_tensor[:, :, 1]))

        loss_meas = loss_measure(output_tensor, input_tensor, measure_v, measure_p, p_norm, Q_norm)
        loss_pde_mass, loss_pde_mom = loss_pde(output_tensor, input_tensor, p_norm, Q_norm,
                                               finite_num=self.finite_number)
        loss_inter = loss_interface(output_tensor, input_tensor, interfaces, p_norm, Q_norm)
        loss = loss_meas + loss_pde_mass + loss_pde_mom + loss_inter

        U, P = output_tensor[:, :, 0], output_tensor[:, :, 1]
        return U, P, (loss, loss_pde_mom, loss_pde_mass, loss_inter, loss_meas)


def loss_measure(output_tensor, input_tensor, measure_v, measure_p, p_norm, Q_norm):
    """

    :param input_tensor: (vessel_num, point_num, [x, are])
    :param output_tensor: (vessel_num, point_num, [u, p])
    :param measure_v: (meas_num, [vessel_id, point_id, value]) 第几个测量点，加在第几个血管的第几个点上
    :param measure_p: (meas_num, [vessel_id, point_id, value])  第几个测量点，加在第几个血管的第几个点上
    :return:
    """
    loss_func = nn.MSELoss(reduction="sum")
    loss_measure_v, loss_measure_p = 0, 0
    for measure_point in measure_v:
        q_pre = output_tensor[measure_point[0], measure_point[1], 0:1] * input_tensor[
            measure_point[0], measure_point[1], 1]
        q_ture = torch.FloatTensor([measure_point[2]]).to(q_pre.device) * input_tensor[
            measure_point[0], measure_point[1], 1]
        q_res = (q_ture - q_pre) / Q_norm
        loss_measure_v += loss_func(q_res, torch.zeros_like(q_res))
    loss_measure_v = loss_measure_v / len(measure_v)  # 对点归一化

    for measure_point in measure_p:
        p_pre = output_tensor[measure_point[0], measure_point[1], 1:]
        p_ture = torch.FloatTensor([measure_point[2]]).to(p_pre.device)
        p_res = (p_ture - p_pre) / p_norm
        loss_measure_p += loss_func(p_res, torch.zeros_like(p_res))
    loss_measure_p = loss_measure_p / len(measure_p)  # 对点归一化

    loss_measure = loss_measure_v + loss_measure_p
    return loss_measure


def loss_interface(output_tensor, input_tensor, interfaces, p_norm, Q_norm, rho=1060):
    """
    :param output_tensor: (vessel_num, point_num, [u, p])
    :param input_tensor: (vessel_num, point_num, [x, are])
    :param interfaces: (interface_num, [father, [son1, ..., son_n]]]) 父亲血管id和儿子血管id
    :param density: 密度
    :return:
    """
    loss_func = nn.MSELoss(reduction="sum")
    loss_interface = 0
    for (father_id, son_ids) in interfaces:
        # ----- 流量守恒 ------
        out_mass = 0
        for son_id in son_ids:
            out_mass += input_tensor[son_id, 0, 1] * output_tensor[son_id, 0, 0]  # 每段入口面积乘上速度
        in_mass = input_tensor[father_id, -1, 1] * output_tensor[father_id, -1, 0]  # 每段出口面积乘上速度
        mass_residual = (in_mass - out_mass) / Q_norm
        loss_mass = loss_func(mass_residual, torch.zeros_like(mass_residual))

        loss_momentum = 0
        # ----- 动量守恒 ------
        # P + ro*U^2/2
        # in_momentum = output_tensor[father_id, -1, 1] + rho * output_tensor[father_id, -1, 0] ** 2 / 2
        in_momentum = output_tensor[father_id, -1, 1] + output_tensor[father_id, -1, 0] ** 2 / 2
        # in_momentum = output_tensor[father_id, -1, 1]
        for son_id in son_ids:
            # out_momentum = output_tensor[son_id, 0, 1] + rho * output_tensor[son_id, 0, 0] ** 2 / 2
            out_momentum = output_tensor[son_id, 0, 1] + output_tensor[son_id, 0, 0] ** 2 / 2
            # out_momentum = output_tensor[son_id, 0, 1]
            momentum_residual = (in_momentum - out_momentum) / p_norm
            loss_momentum += loss_func(momentum_residual, torch.zeros_like(momentum_residual))
        loss_momentum = loss_momentum / len(son_ids)  # 对出口归一化
        loss_interface += loss_mass + loss_momentum
    loss_interface = loss_interface / len(interfaces)
    return loss_interface


def loss_pde(output_tensor, input_tensor, p_norm, Q_norm, rho=1060, mu=0.004, finite_num=20):
    """
        :param output_tensor: (vessel_num, point_num, [u, p])
        :param input_tensor: (vessel_num, point_num, [x, are])
        :param density: 密度
        :param mu:运动粘性
        :param finite_num: simvascular每个segment里面划分有限元个数
        :return:
    """
    # ro = 1060  # 密度
    # mu = 0.004  # 动力粘度
    # nu = mu / ro  # 运动粘度

    vessel_num, point_num, _ = output_tensor.size()
    segment_num = int(point_num // finite_num)  # 血管上血管段个数

    # ----- 动量守恒 ------
    # seg_ids = [20, 40, ..., 380, 400] segment分离点索引
    # last_ids = [19, 39, ..., 379]  每个segment内部，最后一个离散点索引
    # first_ids = [21, ..., 381]  每个segment内部，第一个离散点索引
    seg_ids = torch.tensor([(seg_id + 1) * finite_num for seg_id in range(segment_num)], dtype=torch.long)
    last_ids = seg_ids[:-1] - 1
    first_ids = seg_ids[:-1] + 1

    areas0 = input_tensor[:, last_ids, 1]
    areas1 = input_tensor[:, first_ids, 1]
    Q0 = areas0 * output_tensor[:, last_ids, 0]
    Q1 = areas1 * output_tensor[:, first_ids, 0]
    L = input_tensor[:, seg_ids[:-1] + finite_num, 0] - input_tensor[:, seg_ids[:-1], 0]

    Kt = 1.52
    D0 = 2 * torch.sqrt(areas0 / torch.pi)
    # D1 = 2 * torch.sqrt(areas1 / torch.pi)
    Kv = 32.0 * L / D0 / (areas0 / areas1) ** 2
    Re = output_tensor[:, last_ids, 0] * D0 * rho / mu
    func = Kv / Re + Kt / 2.0 * (areas0 / areas1 - 1.0)
    func_Ns = - 2.0 * areas1 ** 2 * Q0 ** 2 / (areas0 ** 2 * Q1 * L) * func  # 狭窄段N值

    N0 = torch.FloatTensor([-8 * torch.pi * mu / rho]).to(output_tensor.device)
    func_N0 = torch.tile(N0, (vessel_num, 1))  # 正常段N值
    func_N = torch.concat([func_N0, func_Ns], dim=-1)  # (vessel_num, segment_num)
    # func_N = func_N * rho

    loss_pde = 0
    # range_ids = [[0,1,..., 18, 19],[20, 21, ..., 38, 39],...,[380, 381, ..., 398, 399]] 每个segment段21个点的前20点
    range_ids = torch.tensor([[id for id in range(seg_ids[seg_id] - 20, seg_ids[seg_id])]
                              for seg_id in range(segment_num)], dtype=torch.long)
    pressure_in = output_tensor[:, range_ids, 1]
    pressure_out = output_tensor[:, range_ids + 1, 1]
    term1 = pressure_in - pressure_out

    area_in = input_tensor[:, range_ids, 1]
    area_out = input_tensor[:, range_ids + 1, 1]
    Q_in = output_tensor[:, range_ids, 0] * area_in
    Q_out = output_tensor[:, range_ids + 1, 0] * area_out
    # term_2 = 2 / 3 * rho * (output_tensor[:, range_ids, 0] ** 2 - output_tensor[:, range_ids + 1, 0] ** 2)
    term_2 = 2 / 3 * (output_tensor[:, range_ids, 0] ** 2 - output_tensor[:, range_ids + 1, 0] ** 2)

    x_in = input_tensor[:, range_ids, 0]
    x_out = input_tensor[:, range_ids + 1, 0]
    integral_term = (1 / (area_in ** 2) + 1 / (area_out ** 2)) / 2 * (x_out - x_in)  # 积分梯形近似
    term_3 = func_N.unsqueeze(-1) * Q_in * integral_term

    loss_func = nn.MSELoss(reduction="mean")
    loss_residual_momentum = (term1 + term_2 + term_3) / p_norm
    loss_pde_momentum = loss_func(loss_residual_momentum, torch.zeros_like(loss_residual_momentum))

    # ----- 流量守恒 ------
    loss_residual_mass = (Q_out - Q_in) / Q_norm
    loss_pde_mass = loss_func(loss_residual_mass, torch.zeros_like(loss_residual_mass))
    Q_mean = torch.mean(output_tensor[:, :, 0] * input_tensor[:, :, 1], dim=-1)
    loss_residual_mass = (Q_in - Q_mean.unsqueeze(-1).unsqueeze(-1)) / Q_norm
    loss_pde_mass += loss_func(loss_residual_mass, torch.zeros_like(loss_residual_mass))

    return loss_pde_mass, loss_pde_momentum


def data_prepared(finite_number, txt_path):
    with open(txt_path, 'r') as load_f:
        load_dict = json.load(load_f)

    # 每个血管段的x和area
    vessel_number = load_dict["number_vessel"][0]  # 8
    vessel_names = ["vessel_%d" % vessel_id for vessel_id in range(vessel_number)]
    area_names = ["area_%d" % vessel_id for vessel_id in range(vessel_number)]

    vessel_x = [load_dict[vessel_name] for vessel_name in vessel_names]
    vessel_area = [load_dict[area_name] for area_name in area_names]

    vessel_x = np.asarray(vessel_x) / 100  # 8, 21  m
    vessel_area = np.asarray(vessel_area) / 10000  # 8, 21  m^2

    _, vessel_size = vessel_x.shape

    x, area = [], []

    for vessel, vessel_area in zip(vessel_x, vessel_area):
        vessel_x_inter = [np.linspace(vessel[cur], vessel[cur + 1], finite_number, endpoint=False)
                          for cur in range(vessel_size - 1)]  # 400点
        vessel_x_inter = np.concatenate(vessel_x_inter)
        vessel_x_inter = np.concatenate([vessel_x_inter, [vessel[-1]]])

        vessel_area_inter = [np.linspace(vessel_area[cur], vessel_area[cur + 1], finite_number, endpoint=False)
                             for cur in range(vessel_size - 1)]
        vessel_area_inter = np.concatenate(vessel_area_inter)
        vessel_area_inter = np.concatenate([vessel_area_inter, [vessel_area[-1]]])
        x.append(vessel_x_inter)
        area.append(vessel_area_inter)

    # 测量点
    measure_v = [(2, -1, 0.193183), (3, -1, 0.18695), (4, -1, 0.218893), (6, -1, 0.11537), (7, -1, 0.424902)]  # m/s
    measure_p = [(0, 0, 12043 / 1060)]  # pa

    p_norm = measure_p[-1][-1] * 0.01  # 每个血管段误差在0.1%以内
    flow = 0
    for measure_point in measure_v:
        flow += area[measure_point[0]][-1] * measure_point[-1]
    Q_norm = flow * 0.01  # 每个分叉流量误差在1%以内

    # 结点
    interfaces = []
    for interface in load_dict["Link"]:
        father_id = int(interface[0].split("vessel_")[-1])
        son_ids = []
        for son_str in interface[1]:
            son_ids.append(int(son_str.split("vessel_")[-1]))
        interfaces.append((father_id, son_ids))
    return x, area, measure_v, measure_p, interfaces, p_norm, Q_norm


if __name__ == "__main__":

    layers = [2, 100, 100, 100, 2]
    txt_path = "input2.txt"
    finite_number = 20  # 划分有限元
    x, area, measure_v, measure_p, interfaces, p_norm, Q_norm = data_prepared(finite_number, txt_path)

    device = "cpu"
    vessel_number = len(x)

    onepinn = OneDPINNv2(vessel_number=vessel_number, finite_number=finite_number, setting_list=layers, act_func="relu")
    # onepinn.load_state_dict(torch.load("198000_model.pt", map_location=torch.device(device)))
    onepinn.to(device)
    learning_rate = 1e-6
    optimizer = optim.Adam(onepinn.parameters(), lr=learning_rate, betas=(0.95, 0.99), eps=1e-6, weight_decay=1e-6)

    onepinn.train()
    onepinn.zero_grad()
    epochs = 200001

    # x, area, measure_v, measure_p = normalization(x, area, measure_v, measure_p)
    x, area = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device), \
        torch.from_numpy(np.asarray(area, dtype=np.float32)).to(device)
    for epoch in range(epochs):
        u, p, loss_item = onepinn(x, area, measure_v, measure_p, interfaces, p_norm, Q_norm)
        loss_item[0].backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_printf = torch.stack(loss_item).cpu().detach().numpy()
        print(
            'It: %d, Loss: %.3e, Loss_residual_p: %.3e, Loss_residual_Q: %.3e, Loss_interface: %.3e, Loss_measurement: %.3e' %
            (epoch, loss_printf[0], loss_printf[1], loss_printf[2], loss_printf[3], loss_printf[4]))
        if epoch % 20000 == 0 and epoch > 0:
            torch.save(onepinn.state_dict(), "1D/" + str(epoch) + "_model.pt")

        data_p = p.cpu().detach().numpy()
        data_q = u * area / 1e-8
        data_q = data_q.cpu().detach().numpy()
        data_u = u.cpu().detach().numpy()

        U = 10

    u_save = u.cpu().detach().numpy()
    p_save = p.cpu().detach().numpy()
    np.save("u.npy", [u_save])
    np.save("p.npy", [p_save])
