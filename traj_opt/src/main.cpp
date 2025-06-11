#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/frames.hpp>


#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>
#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include <Eigen/Dense>
#include <iostream>
#include <memory>

using namespace pinocchio;
using namespace ifopt;
using namespace Eigen;

// 全局参数
const int N = 10;             // 轨迹点数量
const double dt = 0.02;       // 时间步长
const double total_time = dt * (N-1);  // 总时间

// 机器人模型
Model model;
Data data(model);
pinocchio::container::aligned_vector<pinocchio::Force> f_ext(model.njoints);
int g_nq, g_nv, g_na,g_nf;  // 位置变量数、速度变量数、执行器数量

// VectorXd u_k_full = VectorXd::Zero(model.nv); 

// 初始化机器人模型
void initRobotModel(const std::string& urdf_path) {
    // 加载URDF模型，添加浮动基座
    pinocchio::urdf::buildModel(urdf_path, 
                               JointModelFreeFlyer(), 
                               model);
    // model.gravity.setZero();  // 在轨迹优化中通常忽略重力
    
    // 创建数据对象
    data = Data(model);
    
    // 获取模型维度
    g_nq = model.nq;   // 位置变量数 (浮动基座 + 关节角度)
    g_nv = model.nv;   // 速度变量数
    g_na = g_nv - 6;     // 执行器数量 (减去浮动基座的6个自由度)
    g_nf = 12;

    std::cout << "Robot model loaded: " << urdf_path << "\n";
    std::cout <<"nq: " << g_nq << " nv: " << g_nv << " (Actuated: " << g_na << ")\n";
}

// 轨迹变量集
class StateVariables : public VariableSet {
public:
    StateVariables() : StateVariables("state") {}
    
    StateVariables(const std::string& name) 
        : VariableSet((g_nq + g_nv) * N, name)  // 每个时间点有位置和速度
    {
        // 初始化轨迹为0
        values_ = VectorXd::Ones(GetRows());
    }
    
    void SetVariables(const VectorXd& x) override {
        values_ = x;
    }
    
    VectorXd GetValues() const override {
        return values_;
    }
    
    // 获取特定时间点的状态
    Eigen::VectorBlock<const Eigen::VectorXd> getState(int k) const {
        return values_.segment(k * (g_nq + g_nv), g_nq + g_nv);
    }
    
    // // 设置状态
    // void setState(int k, const VectorXd& state) {
    //     values_.segment(k * (g_nq + g_nv), g_nq + g_nv) = state;
    // }
    
    // 获取位置
    Eigen::VectorBlock<const Eigen::VectorXd> getPosition(int k) const {
        return values_.segment(k * (g_nq + g_nv), g_nq);
    }
    
    // 获取速度
    Eigen::VectorBlock<const Eigen::VectorXd> getVelocity(int k) const {
        return values_.segment(k * (g_nq + g_nv) + g_nq, g_nv);
    }
    
    VecBound GetBounds() const override {
        VecBound bounds(GetRows());
        
        // 设置位置和速度边界
        for (int k = 0; k < N; k++) {
            // 位置边界
            for (int i = 0; i < g_nq; i++) {
                // 浮动基座位置有较大范围，关节角度限制
                if (i < 3) {  // 位置 (x,y,z)
                    bounds[k*(g_nq+g_nv)+i] = Bounds(-10.0, 10.0);
                } else if (i < 7) {  // 四元数方向
                    // 四元数归一化由约束处理
                    bounds[k*(g_nq+g_nv)+i] = NoBound;
                } else {  // 关节角度
                    bounds[k*(g_nq+g_nv)+i] = Bounds(
                        model.lowerPositionLimit[i-7], 
                        model.upperPositionLimit[i-7]
                    );
                }
            }
            
            // 速度边界
            for (int i = 0; i < g_nv; i++) {
                bounds[k*(g_nq+g_nv)+g_nq+i] = Bounds(
                    model.velocityLimit[i] * -1.0,
                    model.velocityLimit[i]
                );
            }
        }
        
        return bounds;
    }

private:
    VectorXd values_;
};

class InputVariables : public VariableSet {
public:
    InputVariables() : InputVariables("input") {}
    
    InputVariables(const std::string& name) 
        : VariableSet((g_na+g_nf) * (N-1), name)  // 每个时间点有g_na个执行器输入
    {
        // 初始化输入为0
        values_ = VectorXd::Zero(GetRows());
    }
    
    void SetVariables(const VectorXd& x) override {
        values_ = x;
    }
    
    VectorXd GetValues() const override {
        return values_;
    }
    
    VecBound GetBounds() const override {
        VecBound bounds(GetRows());
        
        // 设置执行器输入边界 (假设无约束)
        for (int i = 0; i < GetRows(); i++) {
            bounds[i] = NoBound;
        }
        
        return bounds;
    }
    // 获取特定时间点的输入
    Eigen::VectorBlock<const Eigen::VectorXd> getTorqueInput(int k) const {
        return values_.segment(k * (g_na+g_nf), g_na);
    }
    // 获取特定时间点的f输入
    Eigen::VectorBlock<const Eigen::VectorXd> getExtfInput(int k) const {
        return values_.segment(k * (g_na+g_nf) + g_na, g_nf);
    }

private:
    VectorXd values_;
};


// 动力学约束
class DynamicsConstraint : public ConstraintSet {
public:
    DynamicsConstraint() : DynamicsConstraint("dynamics") {}
    
    DynamicsConstraint(const std::string& name)
        : ConstraintSet((g_nv+g_nq) * (N-1), name)  // 每个时间点有nv个约束
    {}
    
    VectorXd GetValues() const override {
        VectorXd g(GetRows());
        auto state_var = std::dynamic_pointer_cast<StateVariables>(GetVariables()->GetComponent("state"));
        auto input_var = std::dynamic_pointer_cast<InputVariables>(GetVariables()->GetComponent("input"));


        // 遍历所有时间点 (k=0 到 k=N-2)
        for (int k = 0; k < N-1; k++) {
            // 获取当前和下一个状态
            VectorXd q_k = state_var->getPosition(k);
            VectorXd v_k = state_var->getVelocity(k);
            VectorXd q_k1 = state_var->getPosition(k+1);
            VectorXd v_k1 = state_var->getVelocity(k+1);
            VectorXd u_k = input_var->getTorqueInput(k);
            VectorXd f_k = input_var->getExtfInput(k);

            // 归一化四元数部分
            q_k.segment<4>(3).normalize();
            q_k1.segment<4>(3).normalize();

            //  // 重置外部力
            for(auto& f : f_ext) f.setZero();

            // 设置局部坐标系的力
            const int lframe_id = model.getFrameId("lleg_link6");
            const int ljoint_id = model.frames[lframe_id].parent;
            f_ext[ljoint_id].linear() = f_k.segment<3>(0);
            f_ext[ljoint_id].angular() = f_k.segment<3>(3);

            const int rframe_id = model.getFrameId("rleg_link6");
            const int rjoint_id = model.frames[rframe_id].parent;
            f_ext[rjoint_id].linear() = f_k.segment<3>(6);
            f_ext[rjoint_id].angular() = f_k.segment<3>(9);

            // // 计算动力学
            VectorXd u_k_full= VectorXd::Zero( model.nv); // 执行器输入向量，前6个为浮动基座的输入

            u_k_full.head(6).setZero();
            u_k_full.tail(g_na) = u_k;  // 填充执行器输入

            q_k.segment<4>(3).normalize();
            q_k1.segment<4>(3).normalize();
            VectorXd a_k = pinocchio::aba(model, data, q_k, v_k, u_k_full,f_ext);
            // VectorXd a_k=VectorXd::Zero(g_nv);
            q_k.segment<4>(3).normalize();
            q_k1.segment<4>(3).normalize();

            // 计算动力学约束(forward euler)
            VectorXd q_diff = q_k1 - pinocchio::integrate(model, q_k, v_k1 * dt);  // 使用Pinocchio的积分器
            // VectorXd q_diff=VectorXd::Zero(g_nq);
            VectorXd v_diff = v_k1 - v_k - a_k * dt;  // 速度差
            // VectorXd v_diff=VectorXd::Zero(g_nv);
            
            // 将位置和速度差存入g
            g.segment(k * (g_nv + g_nq), g_nq) = q_diff;
            g.segment(k * (g_nv + g_nq) + g_nq, g_nv) = v_diff;
        }
        
        return g;
    }
    VecBound GetBounds() const override {
        VecBound bounds(GetRows());
        
        // 设置执行器输入边界 (假设无约束)
        for (int i = 0; i < GetRows(); i++) {
            bounds[i] = Bounds(0.0, 0.0);  // 等式约束
        }
        
        return bounds;
    }
    
    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {}

};

// class QuaternionConstraint : public ifopt::ConstraintSet {
// public:
//     QuaternionConstraint() : ConstraintSet(1, "quaternion_norm") {}
    
//     VectorXd GetValues() const override {
//         VectorXd q = GetVariables()->GetComponent("state")->getPosition();
//         VectorXd value(1);
//         value(0) = q.squaredNorm() - 1.0; // w²+x²+y²+z²-1
//         return value;
//     }
    
//     VecBound GetBounds() const override {
//         VecBound bounds(1);
//         bounds.at(0) = ifopt::Bounds(0.0, 0.0); // 等式约束
//         return bounds;
//     }
    
//     void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override {
//         if (var_set == "quaternion") {
//             VectorXd q = GetVariables()->GetComponent("quaternion")->GetValues();
//             for (int i=0; i<4; ++i) {
//                 jac_block.coeffRef(0, i) = 2.0 * q(i);
//             }
//         }
//     }
// };

// 成本函数
class RefCost : public CostTerm {
public:
    RefCost() : RefCost("reference") {}
    
    RefCost(const std::string& name) : CostTerm(name) {}
    
    double GetCost() const override {
        double cost = 0.0;
        auto state_var = std::dynamic_pointer_cast<StateVariables>(GetVariables()->GetComponent("state"));
        auto input_var = std::dynamic_pointer_cast<InputVariables>(GetVariables()->GetComponent("input"));
        
        //最小化state
        for (int k = 0; k < N; k++) {
            VectorXd state = state_var->getState(k);
            // 计算状态的L2范数
            cost += state.squaredNorm();
        }
        //最小化输入
        for (int k = 0; k < N-1; k++) {
            VectorXd input = input_var->getTorqueInput(k);
            // 计算输入的L2范数
            cost += input.squaredNorm();
        }
        
        return cost;
    }
    
    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {}
};

int main(int argc, char** argv) {
    // if (argc < 2) {
    //     std::cerr << "Usage: " << argv[0] << " <path_to_urdf>" << std::endl;
    //     return 1;
    // }
    
    // 1. 初始化机器人模型
    initRobotModel("/home/bofan/optimization/opt_ws/src/traj_opt/orca_ydescription.urdf");
    
    // 2. 创建问题实例
    Problem nlp;
    
    
    // 创建变量集
    auto state_vars = std::make_shared<StateVariables>();
    auto input_vars = std::make_shared<InputVariables>();

    nlp.AddVariableSet(state_vars);
    nlp.AddVariableSet(input_vars);
    
    // 添加约束
    nlp.AddConstraintSet(std::make_shared<DynamicsConstraint>());
    
    
    // 添加成本函数
    nlp.AddCostSet(std::make_shared<RefCost>());
    
    // 3. 配置求解器
    IpoptSolver ipopt;
    ipopt.SetOption("linear_solver", "mumps");
    ipopt.SetOption("jacobian_approximation", "finite-difference-values");
    ipopt.SetOption("max_iter", 1000);
    ipopt.SetOption("tol", 1e-4);
    ipopt.SetOption("print_level", 5);
    
    // 4. 求解问题
    std::cout << "Solving trajectory optimization problem...\n";
    ipopt.Solve(nlp);
    
    // 5. 提取和显示结果
    VectorXd x_opt = state_vars->GetValues();
    
    // 保存轨迹到文件
    std::ofstream outfile("trajectory.csv");
    outfile << "time,qx,qy,qz,qw";
    for (int i = 0; i < g_na; i++) {
        outfile << ",q" << i;
    }
    outfile << ",vx,vy,vz,wx,wy,wz";
    for (int i = 0; i < g_na; i++) {
        outfile << ",v" << i;
    }
    outfile << "\n";
    
    for (int k = 0; k < N; k++) {
        VectorXd state = x_opt.segment(k*(g_nq+g_nv), g_nq+g_nv);
        VectorXd q = state.head(g_nq);
        VectorXd v = state.tail(g_nv);
        
        outfile << k*dt << ",";
        outfile << q[0] << "," << q[1] << "," << q[2] << "," << q[3];
        for (int i = 7; i < g_nq; i++) {
            outfile << "," << q[i];
        }
        outfile << "," << v[0] << "," << v[1] << "," << v[2] << "," 
                << v[3] << "," << v[4] << "," << v[5];
        for (int i = 6; i < g_nv; i++) {
            outfile << "," << v[i];
        }
        outfile << "\n";
    }
    
    std::cout << "Trajectory saved to trajectory.csv\n";
    std::cout << "Optimization complete!\n";
    
    return 0;
}