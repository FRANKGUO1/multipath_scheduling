from scapy.all import Packet, ByteField, BitField, IntField, get_if_list, bind_layers, Ether


TYPE_PROBE = 0x0812

class probe_header(Packet):
   fields_desc = [ ByteField("num_probe_data", 0)]

class probe_fwd(Packet):
   fields_desc = [ ByteField("src_dst_id", 0)]

class probe_data(Packet):
   fields_desc = [ 
                   BitField("swid", 0, 8),
                #    BitField("ingress_port", 0, 8),
                #    BitField("egress_port", 0, 8),
                   IntField("ingress_byte_cnt", 0),
                   IntField("egress_byte_cnt", 0), 
                   BitField("ingress_last_time",0,48),
                   BitField("ingress_cur_time",0, 48), 
                   BitField("egress_last_time", 0, 48), 
                   BitField("egress_cur_time", 0, 48), 
                   IntField("ingress_packet_count", 0),
                   IntField("egress_packet_count", 0)]

# 获取网卡
def get_if():
    ifs = get_if_list()
    iface = None
    for i in ifs:
        if "eth0" in i:
            iface = i
            break
    if not iface:
        print("Cannot find eth0 interface")
        exit(1)
    return iface

bind_layers(Ether, probe_header, type=TYPE_PROBE)
bind_layers(probe_header, probe_data, data_hop=0)
bind_layers(probe_header, probe_fwd)
bind_layers(probe_fwd, probe_data)
bind_layers(probe_data, probe_data)
bind_layers(probe_data, probe_data)
bind_layers(probe_data, probe_data)