import subprocess


def send_cli_commands(commands, thrift_port):
    """
    向 simple_switch_CLI 发送一系列命令，并返回输出。
    
    :param commands: 要发送的命令列表
    :param thrift_port: Thrift 端口号，默认为 9090
    :return: 每个命令的输出结果
    """
    try:
        # 启动 simple_switch_CLI 进程
        with subprocess.Popen(
            ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
            stdin=subprocess.PIPE,  # 允许写入命令
            stdout=subprocess.PIPE, # 获取命令输出
            stderr=subprocess.PIPE,
            text=True  # 自动处理文本（字符串）
        ) as cli_process:
            outputs = []
            
            for command in commands:
                cli_process.stdin.write(f'{command}\n')
                cli_process.stdin.flush()  # 确保命令已发送
            
            stdout, stderr = cli_process.communicate()
            
            if stderr:
                outputs.append(f"Error: {stderr}")
            else:
                outputs.append(stdout)

            return outputs
    except Exception as e:
        return [f"Failed to run CLI: {e}"]


def run_simple_switch_cli(thrift_port, path_id):
    # 要发送的命令列表
    # 0代表第一条路，1代表第二条路
    commands = [
        f'register_write select_path 0 {path_id}',
        'register_read select_path 0'
    ]
    
    outputs = send_cli_commands(commands, thrift_port)
    
    # for output in outputs:
    #     print(output)