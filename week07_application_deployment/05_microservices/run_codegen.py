from grpc_tools import protoc

protoc.main((
    '',
    '-Iprotos',
    '--python_out=.',
    '--grpc_python_out=.',
    '--pyi_out=.',
    'protos/inference.proto',
))
