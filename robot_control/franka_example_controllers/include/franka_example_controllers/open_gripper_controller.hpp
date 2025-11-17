// open_gripper_controller.hpp
#pragma once

#include <string>
#include <memory>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "franka_msgs/action/move.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * OpenGripperController
 *
 * - Purpose: Open the Franka gripper once and terminate the controller.
 * - Assumes:
 *   - Franka Hand ("Gripper") is attached and ready.
 *   - No repeated toggle or grasping is performed.
 */
class OpenGripperController : public controller_interface::ControllerInterface {
 public:
  OpenGripperController();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

 private:
  bool openGripper();
  void assignMoveGoalOptionsCallbacks();

  rclcpp_action::Client<franka_msgs::action::Move>::SharedPtr gripper_move_action_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr gripper_stop_client_;
  rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions move_goal_options_;

  std::string arm_id_;
  std::string namespace_;
};

}  // namespace franka_example_controllers
