// open_gripper_controller.cpp
// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...

#include <exception>
#include <string>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <franka_msgs/action/move.hpp>

#include "controller_interface/controller_interface.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace franka_example_controllers {

class OpenGripperController : public controller_interface::ControllerInterface {
public:
  controller_interface::InterfaceConfiguration command_interface_configuration() const override {
    return {controller_interface::interface_configuration_type::NONE};
  }
  controller_interface::InterfaceConfiguration state_interface_configuration() const override {
    return {controller_interface::interface_configuration_type::NONE};
  }

  controller_interface::CallbackReturn on_init() override {
    auto_declare<std::string>("arm_id", "fr3");
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State&) override {
    namespace_ = get_node()->get_namespace();

    gripper_move_action_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
        get_node(), namespace_ + "/franka_gripper/move");

    gripper_stop_client_ = get_node()->create_client<std_srvs::srv::Trigger>(
        namespace_ + "/franka_gripper/stop");

    return (gripper_move_action_client_ && gripper_stop_client_)
               ? controller_interface::CallbackReturn::SUCCESS
               : controller_interface::CallbackReturn::ERROR;
  }

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&) override {
    // wait for action server
    if (!gripper_move_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_node()->get_logger(), "Gripper Move action server not available");
      return controller_interface::CallbackReturn::ERROR;
    }

    // 그리퍼 열기
    openGripper();

    

    // 열고 바로 노드 종료
    // on_deactivate(rclcpp_lifecycle::State());
    // std::exit(0);
    rclcpp::shutdown();

    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
    if (gripper_stop_client_->service_is_ready()) {
      auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
      auto result = gripper_stop_client_->async_send_request(request);
      if (result.get() && result.get()->success) {
        RCLCPP_INFO(get_node()->get_logger(), "Gripper stopped successfully.");
        rclcpp::shutdown();
      }
    }
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::return_type update(const rclcpp::Time&, const rclcpp::Duration&) override {
    return controller_interface::return_type::OK;
  }

private:
  void openGripper() {
    franka_msgs::action::Move::Goal move_goal;
    move_goal.width = 0.08;  // fully open
    move_goal.speed = 0.2;

    auto handle = gripper_move_action_client_->async_send_goal(move_goal);
    RCLCPP_INFO(get_node()->get_logger(), "Gripper open command sent");
  }

  std::string namespace_;
  rclcpp_action::Client<franka_msgs::action::Move>::SharedPtr gripper_move_action_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr gripper_stop_client_;
};

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::OpenGripperController,
                       controller_interface::ControllerInterface)
