#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char *TAG = "ESS_ARCHERY";

void app_main(void)
{
    ESP_LOGI(TAG, "=================================");
    ESP_LOGI(TAG, "ESS-Archery V1");
    ESP_LOGI(TAG, "ESP32-C3 boot successful");
    ESP_LOGI(TAG, "=================================");

    while (1)
    {
        ESP_LOGI(TAG, "ESP32-C3 running");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
