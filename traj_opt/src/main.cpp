#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>
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
int g_nq, g_nv, g_na;  // 位置变量数、速度变量数、执行器数量

// 初始化机器人模型
void initRobotModel(const std::string& urdf_path) {
    // 加载URDF模型，添加浮动基座
    pinocchio::urdf::buildModel(urdf_path, 
                               JointModelFreeFlyer(), 
                               model);
    model.gravity.setZero();  // 在轨迹优化中通常忽略重力
    
    // 创建数据对象
    data = Data(model);
    
    // 获取模型维度
    g_nq = model.nq;   // 位置变量数 (浮动基座 + 关节角度)
    g_nv = model.nv;   // 速度变量数
    g_na = g_nv - 6;     // 执行器数量 (减去浮动基座的6个自由度)
    
    std::cout << "Robot model loaded: " << urdf_path << "\n";
    std::cout << "Total DOFs: " << g_nv << " (Actuated: " << g_na << ")\n";
}

// 轨迹变量集
class TrajectoryVariables : public VariableSet {
public:
    TrajectoryVariables() : TrajectoryVariables("trajectory") {}
    
    TrajectoryVariables(const std::string& name) 
        : VariableSet((g_nq + g_nv) * N, name)  // 每个时间点有位置和速度
    {
        // 初始化轨迹为0
        values_ = VectorXd::Zero(GetRows());
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
    
    // 设置状态
    void setState(int k, const VectorXd& state) {
        values_.segment(k * (g_nq + g_nv), g_nq + g_nv) = state;
    }
    
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

// 动力学约束
class DynamicsConstraint : public ConstraintSet {
public:
    DynamicsConstraint() : DynamicsConstraint("dynamics") {}
    
    DynamicsConstraint(const std::string& name)
        : ConstraintSet(g_nv * (N-1), name)  // 每个时间点有nv个约束
    {}
    
    VectorXd GetValues() const override {
        VectorXd g(GetRows());
        auto var = std::dynamic_pointer_cast<TrajectoryVariables>(GetVariables()->GetComponent("trajectory"));
        
        // 遍历所有时间点 (k=0 到 k=N-2)
        for (int k = 0; k < N-1; k++) {
            // 获取当前和下一个状态
            VectorXd q_k = var->getPosition(k);
            VectorXd v_k = var->getVelocity(k);
            VectorXd q_k1 = var->getPosition(k+1);
            VectorXd v_k1 = var->getVelocity(k+1);
            
            // 计算加速度 (v_k1 = v_k + a_k * dt)
            VectorXd a_k = (v_k1 - v_k) / dt;
            
            // 计算逆动力学 (计算力矩)
            VectorXd tau = rnea(model, data, q_k, v_k, a_k);
            
            // 浮动基座没有执行器，所以执行器力矩为0
            tau.head(6).setZero();
            
            // 动力学约束: M*a + C + g = tau
            // 由于浮动基座，前6个元素应为0
            g.segment(k * g_nv, g_nv) = tau;
        }
        
        return g;
    }
    
    VecBound GetBounds() const override {
        VecBound bounds(GetRows());
        
        // 浮动基座动力学约束应为0 (无执行器)
        for (int k = 0; k < N-1; k++) {
            // 前6个自由度 (浮动基座) 必须为0
            for (int i = 0; i < 6; i++) {
                bounds[k*g_nv+i] = Bounds(0.0, 0.0);
            }
            
            // 关节力矩可以有边界 (这里设为无约束)
            for (int i = 6; i < g_nv; i++) {
                bounds[k*g_nv+i] = NoBound;
            }
        }
        
        return bounds;
    }
    
    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
        if (var_set == "trajectory") {
            auto var = std::dynamic_pointer_cast<TrajectoryVariables>(GetVariables()->GetComponent("trajectory"));
            
            // 为雅可比矩阵预留空间
            jac.reserve(g_nv * (N-1) * (g_nq+g_nv) * 2);
            
            // 数值计算雅可比矩阵
            double eps = 1e-6;
            VectorXd x0 = var->GetValues();
            VectorXd g0 = GetValues();
            
            for (int i = 0; i < x0.size(); i++) {
                VectorXd x = x0;
                x(i) += eps;
                var->SetVariables(x);
                VectorXd g_pert = GetValues();
                
                for (int j = 0; j < g0.size(); j++) {
                    double deriv = (g_pert(j) - g0(j)) / eps;
                    if (std::abs(deriv) > 1e-8) {
                        jac.coeffRef(j, i) = deriv;
                    }
                }
            }
            
            // 恢复原始值
            var->SetVariables(x0);
        }
    }
};

// // 初始状态约束
// class InitialStateConstraint : public ConstraintSet {
// public:
//     InitialStateConstraint(const VectorXd& q0, const VectorXd& v0)
//         : InitialStateConstraint("initial_state", q0, v0) {}
    
//     InitialStateConstraint(const std::string& name, 
//                           const VectorXd& q0, const VectorXd& v0)
//         : ConstraintSet(nq + nv, name), q0_(q0), v0_(v0)
//     {}
    
//     VectorXd GetValues() const override {
//         auto var = std::dynamic_pointer_cast<TrajectoryVariables>(GetVariables()->GetComponent("trajectory"));
        
//         VectorXd state = var->getState(0);
//         VectorXd q = state.head(nq);
//         VectorXd v = state.tail(nv);
        
//         VectorXd g(nq + nv);
//         g.head(nq) = q - q0_;
//         g.tail(nv) = v - v0_;
        
//         return g;
//     }
    
//     VecBound GetBounds() const override {
//         VecBound bounds(nq + nv);
//         for (int i = 0; i < nq + nv; i++) {
//             bounds[i] = Bounds(0.0, 0.0);
//         }
//         return bounds;
//     }
    
//     void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
//         if (var_set == "trajectory") {
//             // 只有第一个状态点有非零雅可比
//             for (int i = 0; i < nq + nv; i++) {
//                 jac.coeffRef(i, i) = 1.0;
//             }
//         }
//     }

// private:
//     VectorXd q0_, v0_;
// };

// // 目标位置约束
// class TargetPositionConstraint : public ConstraintSet {
// public:
//     TargetPositionConstraint(int frame_id, const Vector3d& target_pos)
//         : TargetPositionConstraint("target_position", frame_id, target_pos) {}
    
//     TargetPositionConstraint(const std::string& name, 
//                             int frame_id, const Vector3d& target_pos)
//         : ConstraintSet(3, name), frame_id_(frame_id), target_pos_(target_pos)
//     {}
    
//     VectorXd GetValues() const override {
//         auto var = std::dynamic_pointer_cast<TrajectoryVariables>(GetVariables()->GetComponent("trajectory"));
        
//         // 获取最终位置
//         VectorXd q_final = var->getPosition(N-1);
        
//         // 计算末端执行器位置
//         updateFramePlacement(model, data, frame_id_);
//         Vector3d pos = data.oMf[frame_id_].translation();
        
//         return pos - target_pos_;
//     }
    
//     VecBound GetBounds() const override {
//         VecBound bounds(3);
//         for (int i = 0; i < 3; i++) {
//             bounds[i] = Bounds(0.0, 0.0);
//         }
//         return bounds;
//     }

// private:
//     int frame_id_;
//     Vector3d target_pos_;
// };

// 平滑性成本函数
class SmoothnessCost : public CostTerm {
public:
    SmoothnessCost() : SmoothnessCost("smoothness") {}
    
    SmoothnessCost(const std::string& name) : CostTerm(name) {}
    
    double GetCost() const override {
        double cost = 0.0;
        auto var = std::dynamic_pointer_cast<TrajectoryVariables>(GetVariables()->GetComponent("trajectory"));
        
        // 最小化加速度
        for (int k = 0; k < N-1; k++) {
            VectorXd v_k = var->getVelocity(k);
            VectorXd v_k1 = var->getVelocity(k+1);
            VectorXd a_k = (v_k1 - v_k) / dt;
            
            cost += a_k.squaredNorm();
        }
        
        // 最小化力矩变化
        for (int k = 1; k < N-1; k++) {
            // 计算力矩 (简化的)
            VectorXd q_k = var->getPosition(k);
            VectorXd v_k = var->getVelocity(k);
            VectorXd v_k1 = var->getVelocity(k+1);
            VectorXd a_k = (v_k1 - v_k) / dt;
            
            VectorXd tau_k = rnea(model, data, q_k, v_k, a_k).tail(g_na);
            
            // 上一个时间点的力矩
            VectorXd q_km1 = var->getPosition(k-1);
            VectorXd v_km1 = var->getVelocity(k-1);
            VectorXd a_km1 = (v_k - v_km1) / dt;
            VectorXd tau_km1 = rnea(model, data, q_km1, v_km1, a_km1).tail(g_na);
            
            cost += (tau_k - tau_km1).squaredNorm();
        }
        
        return cost;
    }
    
    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
        // 数值计算梯度
        if (var_set == "trajectory") {
            double eps = 1e-6;
            auto var = std::dynamic_pointer_cast<TrajectoryVariables>(GetVariables()->GetComponent("trajectory"));
            
            VectorXd x0 = var->GetValues();
            double cost0 = GetCost();
            
            for (int i = 0; i < x0.size(); i++) {
                VectorXd x = x0;
                x(i) += eps;
                var->SetVariables(x);
                double cost_pert = GetCost();
                
                jac.coeffRef(0, i) = (cost_pert - cost0) / eps;
            }
            
            // 恢复原始值
            var->SetVariables(x0);
        }
    }
};

int main(int argc, char** argv) {
    // if (argc < 2) {
    //     std::cerr << "Usage: " << argv[0] << " <path_to_urdf>" << std::endl;
    //     return 1;
    // }
    
    // 1. 初始化机器人模型
    initRobotModel("/home/bofan/optimization/URDFopt/orca_ydescription.urdf");
    
    // 2. 创建问题实例
    Problem nlp;
    
    // 初始状态 (零状态)
    VectorXd q0 = VectorXd::Zero(g_nq);
    VectorXd v0 = VectorXd::Zero(g_nv);
    
    // 设置浮动基座初始位置和方向
    q0.head<3>().setZero();        // 位置
    q0.segment<4>(3) << 0, 0, 0, 1; // 四元数 (x,y,z,w)
    
    // 创建变量集
    auto trajectory_vars = std::make_shared<TrajectoryVariables>();
    nlp.AddVariableSet(trajectory_vars);
    
    // 添加约束
    // nlp.AddConstraintSet(std::make_shared<InitialStateConstraint>(q0, v0));
    nlp.AddConstraintSet(std::make_shared<DynamicsConstraint>());
    
    // // 添加末端执行器目标约束 (示例：使用第一个操作链的末端)
    // int target_frame_id = 0;
    // for (const auto& frame : model.frames) {
    //     if (frame.name.find("hand") != std::string::npos) {
    //         target_frame_id = frame.id;
    //         std::cout << "Using target frame: " << frame.name << "\n";
    //         break;
    //     }
    // }
    
    // Vector3d target_pos(1.0, 0.5, 0.5);  // 目标位置
    // nlp.AddConstraintSet(std::make_shared<TargetPositionConstraint>(
    //     target_frame_id, target_pos));
    
    // 添加成本函数
    nlp.AddCostSet(std::make_shared<SmoothnessCost>());
    
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
    VectorXd x_opt = trajectory_vars->GetValues();
    
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