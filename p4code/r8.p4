/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

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
register< bit<32> >(1) select_path;

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
// ICMP首部
header icmp_h {
    bit<8>   type;
    bit<8>   code;
    bit<16>  hdr_checksum;
}
//--------------------------

//--------------------------
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
//--------------------------
//UDP首部
header udp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<16>  hdr_length;
    bit<16>  checksum;
}
//--------------------------
struct metadata {
    bit<32> path_id;
}

struct headers {
    ethernet_h   ethernet;
    arp_h        arp;
    ipv4_t       ipv4;
    icmp_h       icmp;
    tcp_h        tcp;
    udp_h        udp;
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        meta = {0};
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type){
            TYPE_ARP: parse_arp;
            TYPE_IPV4: ipv4;
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
    apply {
        if (hdr.arp.isValid()) {
            // arp欺骗
            // is the packet for arp
            if (hdr.arp.target_ip == 0x0a080101) {
                //ask who is 10.2.1.1
                hdr.ethernet.dst_mac = hdr.ethernet.src_mac;
                hdr.ethernet.src_mac = 0x00000a080102;
                hdr.arp.OPER = 2;
                hdr.arp.target_ha = hdr.arp.sender_ha;
                hdr.arp.target_ip = hdr.arp.sender_ip;
                hdr.arp.sender_ip = 0x0a080101;
                hdr.arp.sender_ha = 0x00000a080102;
                standard_metadata.egress_spec = standard_metadata.ingress_port;
            }
        }
        // Only if IPV4 the rule is applied. Therefore other packets will not be forwarded.
        else if (hdr.ipv4.isValid()){
            bit<32> temp = 0;
            select_path.read(temp, (bit<32>)0);
            meta.path_id = temp;
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

   apply {
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