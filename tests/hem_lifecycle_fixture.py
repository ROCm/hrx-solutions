"""Shared CPU lifecycle fixtures; no test-case modules are imported here."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

from binding.python import hem_process
from binding.python.hem_common import UncertainOutcome

def inventory(*items):
    return {"object": "list", "data": list(items), "server": {"worker_count": 1}}


def model(incarnation=17, **overrides):
    loom = dict(incarnation=incarnation, registration="ready", parameter_operation="idle",
                parameters={"active": True}, status={"admission": "drained"})
    loom.update(overrides)
    return dict(id="chat", loom=loom)


@contextmanager
def fixture(respond=None):
    records = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

        def dispatch(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            records.append((self.command, self.path, body, dict(self.headers)))
            if respond is not None:
                try:
                    respond(self, records)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                reply(self, inventory(model()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.005), daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port), records
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def reply(handler, payload, status=200):
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_exit(process, seconds=3):
    until = time.monotonic() + seconds
    uncertain = None
    while time.monotonic() < until:
        try:
            if hem_process._confirmed_exit(process):
                return
        except UncertainOutcome as error:
            # Procfs executable data may disappear before pidfd reports exit.
            # Retry the same owned record; uncertainty never counts as success.
            uncertain = error
        time.sleep(.01)
    if uncertain is not None:
        raise uncertain
    raise AssertionError(f'owned process did not exit: {process}')


# An independent ELF proves bootstrap-to-native executable identity.
PROCESS_NATIVE = r'''
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc, char **argv) {
 signal(SIGPIPE,SIG_IGN);
 int port=0, fd, client, yes=1; char request[2048];
 int unhealthy=0;
 for(int i=1;i<argc;i++) if(!strncmp(argv[i],"--listen=",9)) port=atoi(strrchr(argv[i], ':')+1);
 for(int i=1;i<argc;i++) { if(!strcmp(argv[i],"--device=slow")) usleep(500000); if(!strcmp(argv[i],"--device=unhealthy")) unhealthy=1; }
 if(!port) return 5;
 fd=socket(AF_INET,SOCK_STREAM,0); setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
 struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(port),.sin_addr.s_addr=htonl(INADDR_LOOPBACK)};
 if(bind(fd,(void*)&a,sizeof(a)) || listen(fd,16)) return 6;
 fprintf(stderr,"fake-native ready %d\n",port); fflush(stderr);
 while((client=accept(fd,NULL,NULL))>=0) {
   int count=read(client,request,sizeof(request)-1); if(count<0) count=0; request[count]=0;
   const char *body="{\"object\":\"list\",\"data\":[],\"server\":{\"worker_count\":1}}";
   if(strstr(request,"/noise")) { for(int i=0;i<100000;i++) fputs("bounded diagnostic output 1234567890\n",stderr); fflush(stderr); }
   if(unhealthy || strstr(request,"/fail")) body="{}";
   char reply[4096]; int n=snprintf(reply,sizeof(reply),"HTTP/1.1 200 OK\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",strlen(body),body);
   write(client,reply,n); close(client);
 }
 return 0;
}
'''


# Real pin/accepted-connection association through registration and retirement.
INSTANCES_NATIVE = r'''
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc, char **argv) {
 signal(SIGPIPE,SIG_IGN);
 int port=0, fd, client, yes=1, registered=0, active=0, open=0;
 for(int i=1;i<argc;i++) if(!strncmp(argv[i],"--listen=",9)) port=atoi(strrchr(argv[i], ':')+1);
 if(!port) return 5;
 fd=socket(AF_INET,SOCK_STREAM,0); setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
 struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(port),.sin_addr.s_addr=htonl(INADDR_LOOPBACK)};
 if(bind(fd,(void*)&a,sizeof(a)) || listen(fd,16)) return 6;
 while((client=accept(fd,NULL,NULL))>=0) {
   char request[65536]={0}, body[8192], response[10000]; int count=0, n=0;
   while(count < (int)sizeof(request)-1 && (n=read(client,request+count,sizeof(request)-count-1))>0) {
     count+=n; request[count]=0; char *end=strstr(request,"\r\n\r\n");
     if(end) { char *cl=strstr(request,"Content-Length:"); int length=cl ? atoi(cl+15) : 0;
       if(count >= (int)(end-request)+4+length) break; }
   }
   int code=200;
   if(strstr(request,"POST /v1/models/register ")) { registered=1; active=0; open=0; code=202; }
   if(strstr(request,"POST /v1/models/activate ")) { active=1; code=202; }
   if(strstr(request,"POST /v1/models/undrain ")) { if(!active) return 7; open=1; code=202; }
   if(strstr(request,"POST /v1/models/unregister ")) { registered=0; code=202; }
   if(!strncmp(request,"POST",4)) strcpy(body,"{\"accepted\":true}");
   else if(registered && !active) strcpy(body,"{\"object\":\"list\",\"data\":[{\"id\":\"chat\",\"loom\":{\"incarnation\":17,\"registration\":\"registered\",\"parameter_operation\":\"idle\",\"status\":{\"admission\":\"drained\"}}}],\"server\":{\"worker_count\":1}}");
   else if(registered) snprintf(body,sizeof(body),"{\"object\":\"list\",\"data\":[{\"id\":\"chat\",\"loom\":{\"incarnation\":17,\"registration\":\"ready\",\"parameter_operation\":\"idle\",\"parameters\":{\"active\":%s},\"status\":{\"admission\":\"%s\"}}}],\"server\":{\"worker_count\":1}}",active?"true":"false",open?"open":"drained");
   else strcpy(body,"{\"object\":\"list\",\"data\":[],\"server\":{\"worker_count\":1}}");
   n=snprintf(response,sizeof(response),"HTTP/1.1 %d OK\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",code,strlen(body),body);
   write(client,response,n); close(client);
 }
 return 0;
}
'''


# Public server CLI diagnostics and health protocol.
SERVER_CLI_NATIVE = r'''
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc,char **argv) {
 signal(SIGPIPE,SIG_IGN); int port=0,yes=1,unhealthy=0;
 for(int i=1;i<argc;i++) if(!strncmp(argv[i],"--listen=",9)) port=atoi(strrchr(argv[i],':')+1);
 int s=socket(AF_INET,SOCK_STREAM,0); setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
 struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(port),.sin_addr.s_addr=htonl(INADDR_LOOPBACK)};
 if(!port || bind(s,(void*)&a,sizeof(a)) || listen(s,16)) return 6;
 fprintf(stderr,"fake-native ready %d\n",port); fflush(stderr);
 int c; while((c=accept(s,NULL,NULL))>=0) {
  char request[4096]; int n=read(c,request,sizeof(request)-1); if(n<0)n=0; request[n]=0;
  fprintf(stderr,"%.*s\n",(int)strcspn(request,"\r\n"),request); fflush(stderr);
  if(strstr(request,"/fail")) unhealthy=1;
  if(strstr(request,"/noise")) {for(int i=0;i<100000;i++) fputs("bounded fixture log output abcdefghijklmnopqrstuvwxyz\n",stderr); fflush(stderr);}
  const char *body=unhealthy?"{}":"{\"object\":\"list\",\"data\":[],\"server\":{\"worker_count\":1}}";
  dprintf(c,"HTTP/1.1 200 OK\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",strlen(body),body); close(c);
 }
 return 0;
}
'''
