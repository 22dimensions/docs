# builder

1. docker image name format (registry)
2. our registry organization --- and config
3. our base image
4. space server ----(build request)---> docker builder ---(push)-----> registry <------(pull)--------driver

 - https://docs.docker.com/reference/dockerfile/#cmd
 - base image + tpl
 - text/template 文本生成 (https://golangdocs.com/templates-in-golang)
 - https://docs.docker.com/glossary/#registry
 - https://docs.docker.com/docker-hub/mirror/#configure-the-docker-daemon
 - If CMD is used to provide default arguments for the ENTRYPOINT instruction, both the CMD and ENTRYPOINT instructions should be specified in the exec form.
