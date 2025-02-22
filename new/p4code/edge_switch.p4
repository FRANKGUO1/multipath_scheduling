/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

#define MAX_PORTS 255
#define MAX_HOPS 5

const bit<48> VIRTUAL_MAC = 0x000000000001;

const bit<16> TYPE_ARP = 0x0806;
const bit<16> TYPE_PROBE = 0x0812;
const bit<16> TYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_ICMP = 0x01;
const bit<8>  IP_PROTO_TCP = 0x06;
const bit<8>  IP_PROTO_UDP = 0x11;

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

//------------------------------------------------------------
// 定义首部
// 物理层首部
register< bit<8> >(1) select_path;

header ethernet_h {
    bit<48>  dst_mac;
    bit<48>  src_mac;
    bit<16>  ether_type;
}
//--------------------------
// ARP首部
header arp_h {
    bit<16>  hardware_type;
    bit<16>  protocol_type;
    bit<8>   HLEN;
    bit<8>   PLEN;
    bit<16>  OPER;
    bit<48>  sender_ha;
    bit<32>  sender_ip;
    bit<48>  target_ha;
    bit<32>  target_ip;
}

header probe_header_t {
    bit<8> num_probe_data;    //记录这个探测包已经通过了几个交换机
}

header probe_fwd_h {
    bit<8>   src_dst_id; // 交换机路径标识
}

header probe_data_t {
    bit<8>    swid;     
    // bit<8>    ingress_port;
    // bit<8>    egress_port;
    bit<32>   ingress_byte_cnt;
    bit<32>   egress_byte_cnt;
    bit<48>    ingress_last_time;
    bit<48>    ingress_cur_time;        //有些数据不用记录，但是为了看上去对称就都写了
    bit<48>    egress_last_time;
    bit<48>    egress_cur_time;
    bit<32>    ingress_packet_count;
    bit<32>    egress_packet_count;
}
//--------------------------

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    tos;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

//------------ICMP首部-------------- 
header icmp_h {
    bit<8>   type;
    bit<8>   code;
    bit<16>  hdr_checksum;
}
//---------------------------------

//---------------------------------
//TCP首部
header tcp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<32>  seq_no;
    bit<32>  ack_no;
    bit<4>   data_offset;
    bit<4>   res;
    bit<8>   flags;
    bit<16>  window;
    bit<16>  checksum;
    bit<16>  urgent_ptr;
}
//---------------------------------
//UDP首部
header udp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<16>  hdr_length;
    bit<16>  checksum;
}
//---------------------------------
struct metadata {
    bit<8>  path_id;
    bit<8>   remaining1;
    bit<8>   remaining2;
    bit<8>   sswid;
    bit<32>  pktcont2;
    bit<9>   ingress_time;
    bit<8>   segment_left;
    bit<128> segment_id; 
}

struct headers {
    ethernet_h               ethernet;
    probe_header_t           probe_header;
    probe_fwd_h              probe_fwd;
    probe_data_t[MAX_HOPS]   probe_data;
    arp_h                    arp;
    ipv4_t                   ipv4;
    icmp_h                   icmp;
    tcp_h                    tcp;
    udp_h                    udp;   
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        meta = {0, 0, 0, 0, 0, 0, 0, 0};
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type){
            TYPE_ARP: parse_arp;
            TYPE_IPV4: ipv4;
            TYPE_PROBE: parse_probe;  //说明这个包是一个探测包，而且刚刚从发送端出来
            default: accept;
        }

    }

    state parse_arp {
        packet.extract(hdr.arp);
        transition accept;
    }

    state ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IP_PROTO_ICMP: parse_icmp;
            IP_PROTO_TCP: parse_tcp;
            IP_PROTO_UDP: parse_udp;   
            default: accept;
        }
    }

    state parse_probe {
        packet.extract(hdr.probe_header);
        meta.remaining1=hdr.probe_header.num_probe_data;
        transition parse_probe_fwd_h;
    }

    state parse_probe_fwd_h {
        packet.extract(hdr.probe_fwd);
        transition select(meta.remaining1) {
            0: accept;
            default:parse_probe_list;
        }
    }

    state parse_probe_list{
        packet.extract(hdr.probe_data.next);
        meta.remaining1=meta.remaining1-1;
        transition select(meta.remaining1){
            0:accept;
            default: parse_probe_list;
        }
    }

    state parse_icmp {
        packet.extract(hdr.icmp);
        transition accept;
    }

    state parse_tcp {
       packet.extract(hdr.tcp);
       transition accept;
    }

    state parse_udp {
       packet.extract(hdr.udp);
       transition accept;
    }

}


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
    register<bit<32>>(MAX_PORTS) byte_cnt_reg;
    register<bit<32>>(MAX_PORTS) packet_cnt_reg;
    register<bit<48>>(MAX_PORTS) last_time_reg;

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(bit<48> dst_mac, egressSpec_t port) {
        // Set the output port that we also get from the table
        hdr.ethernet.src_mac = hdr.ethernet.dst_mac;
        hdr.ethernet.dst_mac = dst_mac;
        standard_metadata.egress_spec = port;
        // Decrease ttl by 1
        hdr.ipv4.ttl = hdr.ipv4.ttl -1;
    }

    table ipv4_lpm {
        key = {
            meta.path_id: exact;
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    action probe_forward(bit<48> dst_mac, bit<9> port) {
        hdr.ethernet.src_mac = hdr.ethernet.dst_mac;
        hdr.ethernet.dst_mac = dst_mac;
        standard_metadata.egress_spec = port;
    }

    table probe_exact {
        key = {
            // meta.path_id: exact;
            hdr.probe_fwd.src_dst_id: exact;       
        }
        actions = {
            probe_forward;
        }
        size = 1024;
    }

    apply {
        bit<32> packet_cnt;
        bit<32> new_packet_cnt;
        bit<32> byte_cnt;
        bit<32> new_byte_cnt;
        bit<48> last_time;
        bit<48> cur_time = standard_metadata.ingress_global_timestamp;
        bit<8> temp = 0;
        select_path.read(temp, (bit<32>)0);
        meta.path_id = temp;
        byte_cnt_reg.read(byte_cnt, (bit<32>)standard_metadata.ingress_port);
        // 这里总感觉有问题，为何要给new_byte_cnt置零，吞吐量的计算方式？
        byte_cnt = byte_cnt + standard_metadata.packet_length;
        new_byte_cnt = (hdr.probe_header.isValid()) ? 0 : byte_cnt; // 这里就是不计入Int的数据包长度，如果是Int包，则置0
        byte_cnt_reg.write((bit<32>)standard_metadata.ingress_port, new_byte_cnt); // 计算两个Int包之间的吞吐量
        
        packet_cnt_reg.read(packet_cnt, (bit<32>)standard_metadata.ingress_port);
        packet_cnt = packet_cnt + 1;
        new_packet_cnt = (hdr.probe_header.isValid()) ? 0 : packet_cnt;
        packet_cnt_reg.write((bit<32>)standard_metadata.ingress_port, new_packet_cnt);

        if(hdr.probe_header.isValid()) {
            hdr.probe_data.push_front(1);
            hdr.probe_data[0].setValid();    // 说明这就是一个INT包
            hdr.probe_header.num_probe_data=hdr.probe_header.num_probe_data + 1; // probe_data个数加1
            hdr.probe_data[0].swid = hdr.probe_fwd.src_dst_id;
            // swid.apply();
            // hdr.probe_data[0].ingress_port = (bit<8>)standard_metadata.ingress_port;
            hdr.probe_data[0].ingress_byte_cnt = byte_cnt;

            last_time_reg.read(last_time, (bit<32>)standard_metadata.ingress_port);
            last_time_reg.write((bit<32>)standard_metadata.ingress_port, cur_time);
            hdr.probe_data[0].ingress_last_time = last_time;
            hdr.probe_data[0].ingress_cur_time = cur_time;
            hdr.probe_data[0].ingress_packet_count = packet_cnt;
  
            probe_exact.apply();
        }

        if (hdr.arp.isValid()) {
            // arp欺骗
            // is the packet for arp
            if (hdr.arp.target_ip == 0x0a010101) {
                //ask who is 10.1.1.1
                hdr.ethernet.dst_mac = hdr.ethernet.src_mac;
                hdr.ethernet.src_mac = 0x00000a010102;
                hdr.arp.OPER = 2;
                hdr.arp.target_ha = hdr.arp.sender_ha;
                hdr.arp.target_ip = hdr.arp.sender_ip;
                hdr.arp.sender_ip = 0x0a010101;
                hdr.arp.sender_ha = 0x00000a010102;
                standard_metadata.egress_spec = standard_metadata.ingress_port;
            }
            else if (hdr.arp.target_ip == 0x0a020101) {
                //ask who is 10.2.1.1
                hdr.ethernet.dst_mac = hdr.ethernet.src_mac;
                hdr.ethernet.src_mac = 0x00000a020102;
                hdr.arp.OPER = 2;
                hdr.arp.target_ha = hdr.arp.sender_ha;
                hdr.arp.target_ip = hdr.arp.sender_ip;
                hdr.arp.sender_ip = 0x0a020101;
                hdr.arp.sender_ha = 0x00000a020102;
                standard_metadata.egress_spec = standard_metadata.ingress_port;
            }
            else if (hdr.arp.target_ip == 0x0a030101) {
                //ask who is 10.3.1.1
                hdr.ethernet.dst_mac = hdr.ethernet.src_mac;
                hdr.ethernet.src_mac = 0x00000a030102;
                hdr.arp.OPER = 2;
                hdr.arp.target_ha = hdr.arp.sender_ha;
                hdr.arp.target_ip = hdr.arp.sender_ip;
                hdr.arp.sender_ip = 0x0a030101;
                hdr.arp.sender_ha = 0x00000a030102;
                standard_metadata.egress_spec = standard_metadata.ingress_port;
            }
            else if (hdr.arp.target_ip == 0x0a080101) {
                //ask who is 10.8.1.1
                hdr.ethernet.dst_mac = hdr.ethernet.src_mac;
                hdr.ethernet.src_mac = 0x00000a080102;
                hdr.arp.OPER = 2;
                hdr.arp.target_ha = hdr.arp.sender_ha;
                hdr.arp.target_ip = hdr.arp.sender_ip;
                hdr.arp.sender_ip = 0x0a080101;
                hdr.arp.sender_ha = 0x00000a080102;
                standard_metadata.egress_spec = standard_metadata.ingress_port;
            }
            else if (hdr.arp.target_ip == 0x0a090101) {
                //ask who is 10.9.1.1
                hdr.ethernet.dst_mac = hdr.ethernet.src_mac;
                hdr.ethernet.src_mac = 0x00000a090102;
                hdr.arp.OPER = 2;
                hdr.arp.target_ha = hdr.arp.sender_ha;
                hdr.arp.target_ip = hdr.arp.sender_ip;
                hdr.arp.sender_ip = 0x0a090101;
                hdr.arp.sender_ha = 0x00000a090102;
                standard_metadata.egress_spec = standard_metadata.ingress_port;
            }
            else if (hdr.arp.target_ip == 0x0a0a0101) {
                //ask who is 10.10.1.1
                hdr.ethernet.dst_mac = hdr.ethernet.src_mac;
                hdr.ethernet.src_mac = 0x00000a0a0102;
                hdr.arp.OPER = 2;
                hdr.arp.target_ha = hdr.arp.sender_ha;
                hdr.arp.target_ip = hdr.arp.sender_ip;
                hdr.arp.sender_ip = 0x0a0a0101;
                hdr.arp.sender_ha = 0x00000a0a0102;
                standard_metadata.egress_spec = standard_metadata.ingress_port;
            }
        }
        // Only if IPV4 the rule is applied. Therefore other packets will not be forwarded.
        else if (hdr.ipv4.isValid()){
            ipv4_lpm.apply();
        }
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    
    register<bit<32>>(MAX_PORTS) byte_cnt_reg;
    register<bit<32>>(MAX_PORTS) packet_cnt_reg;
    register<bit<48>>(MAX_PORTS) last_time_reg;

    apply {
        bit<32> packet_cnt;
        bit<32> new_packet_cnt;
        bit<32> byte_cnt;
        bit<32> new_byte_cnt;
        bit<48> last_time;
        bit<48> cur_time = standard_metadata.egress_global_timestamp;
        byte_cnt_reg.read(byte_cnt, (bit<32>)standard_metadata.egress_port);
        byte_cnt = byte_cnt + standard_metadata.packet_length;
        new_byte_cnt = (hdr.probe_header.isValid()) ? 0 : byte_cnt;
        byte_cnt_reg.write((bit<32>)standard_metadata.egress_port, new_byte_cnt);
        
        packet_cnt_reg.read(packet_cnt, (bit<32>)standard_metadata.egress_port);
        packet_cnt = packet_cnt + 1;
        new_packet_cnt = (hdr.probe_header.isValid()) ? 0 : packet_cnt;
        packet_cnt_reg.write((bit<32>)standard_metadata.egress_port, new_packet_cnt);

        if(hdr.probe_header.isValid()){
            // hdr.probe_data[0].egress_port = (bit<8>)standard_metadata.egress_port;
            hdr.probe_data[0].egress_byte_cnt = byte_cnt;

            last_time_reg.read(last_time, (bit<32>)standard_metadata.egress_port);
            last_time_reg.write((bit<32>)standard_metadata.egress_port, cur_time);
            hdr.probe_data[0].egress_last_time = last_time;
            hdr.probe_data[0].egress_cur_time = cur_time;
            hdr.probe_data[0].egress_packet_count = packet_cnt;
           }
        }
    }


/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	          hdr.ipv4.ihl,
              hdr.ipv4.tos,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
              hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}


/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {

        packet.emit(hdr.ethernet);
        packet.emit(hdr.probe_header);
        packet.emit(hdr.probe_fwd);
        packet.emit(hdr.probe_data);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.arp);
        packet.emit(hdr.icmp);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

//switch architecture
V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;