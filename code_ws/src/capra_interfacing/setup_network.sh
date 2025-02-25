#!/bin/bash

# Set static IP
interface eth0
static ip_address=10.46.28.4/24
static routers=10.46.28.1
static domain_name_servers=10.46.28.1" >> /etc/dhcpcd.conf 
