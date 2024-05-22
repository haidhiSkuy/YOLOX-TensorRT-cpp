#include <signal.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <hiredis/hiredis.h>
#include <hiredis/async.h>
#include <string.h> 
#include <hiredis/adapters/libevent.h>


int main (int argc, char **argv) {
    redisContext *c = redisConnect("localhost", 6379);
    if (c->err) {
        printf("error: %s\n", c->errstr);
        return 1;
    }   

    const std::string channel = "test_channel";
    const std::string message = "Coba C++";

    redisReply* reply = (redisReply*)redisCommand(c, "PUBLISH %s %s", channel.c_str(), message.c_str());

}

