#include <iostream>
#include "handle.h"
using namespace std;
using namespace HttpServer;
int main(void){
    
    httplib::Server svr;

    svr.Post("/yolo/v1", handleProcessImage);

    if (svr.listen("0.0.0.0", 12124)) {
        std::cout << "Server is running at http://localhost:12124" << std::endl;
    } else {
        std::cerr << "Failed to start server." << std::endl;
        return 1;
    }
    return 0;
}