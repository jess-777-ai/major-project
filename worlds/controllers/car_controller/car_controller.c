#include <webots/robot.h>
#include <webots/supervisor.h>

int main()
{
  wb_robot_init();
  int time_step = wb_robot_get_basic_time_step();

  while (wb_robot_step(time_step) != -1)
  {
    wb_supervisor_set_label(
        0,
        "✅ Simulation Running - Model Ready",
        0.01, 0.05,
        0.07,
        0xFFFFFF,
        0.0,
        "Arial");
  }

  wb_robot_cleanup();
  return 0;
}
