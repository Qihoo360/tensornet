// Copyright (c) 2020, Qihoo, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/utility/net_util.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace tensornet {

std::string get_local_ip_internal() {
    int sockfd = -1;
    char buf[512];
    struct ifconf ifconf;
    struct ifreq* ifreq;

    ifconf.ifc_len = 512;
    ifconf.ifc_buf = buf;

    if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        return "";
    }

    if(ioctl(sockfd, SIOCGIFCONF, &ifconf) < 0) {
        return "";
    }

    if(0 != close(sockfd)) {
        return "";
    }

    ifreq = (struct ifreq*)buf;
    for (int i = 0; i < int(ifconf.ifc_len / sizeof(struct ifreq)); ++i) {
        std::string ip;
        ip = inet_ntoa(((struct sockaddr_in*)&ifreq->ifr_addr)->sin_addr);
        if (ip != "127.0.0.1") {
            return ip;
        }

        ifreq++;
    }

    return "";
}

static uint16_t get_useable_port_aux() {
    struct sockaddr_in addr;
    addr.sin_port = htons(0);  // have system pick up a random port available for me
    addr.sin_family = AF_INET;  // IPV4
    addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set our addr to any interface

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    socklen_t addr_len = sizeof(struct sockaddr_in);
    if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
        perror("get socket name fail");
        exit(EXIT_FAILURE);
    }

    int ret_port = ntohs(addr.sin_port);
    close(sock);
    return ret_port;
}

uint16_t get_useable_port() {
    uint16_t port = 0;

    while (port < 1024) {
        port = get_useable_port_aux();
    }

    return port;
}

} // namespace tensornet
