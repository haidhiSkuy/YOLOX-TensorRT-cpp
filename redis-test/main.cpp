#include <signal.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <hiredis/hiredis.h>
#include <hiredis/async.h>
#include <string.h> 
#include <hiredis/adapters/libevent.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>


nlohmann::json rectToJson(const cv::Rect& rect) {
    nlohmann::json j;
    j["x"] = rect.x;
    j["y"] = rect.y;
    j["width"] = rect.width;
    j["height"] = rect.height;
    return j;
} 


int main (int argc, char **argv) {
    redisContext *c = redisConnect("localhost", 6379);
    if (c->err) {
        printf("error: %s\n", c->errstr);
        return 1;
    }   

    const std::string channel = "test_channel";

    cv::Rect rect(10, 20, 100, 200);
    nlohmann::json rectJson = rectToJson(rect);
    std::string jsonString = rectJson.dump();

    redisReply* reply = (redisReply*)redisCommand(c, "PUBLISH %s %s", channel.c_str(), jsonString.c_str()); 
}

