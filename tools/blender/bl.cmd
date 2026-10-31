@echo off
rem Windows counterpart of bl.sh: run Blender with every user, cache and temp path inside
rem third_party/blender/ (this folder: the Blender build, user/ with the VRM Add-on, home/, tmp/).
set "B=%~dp0"
if "%B:~-1%"=="\" set "B=%B:~0,-1%"
if not exist "%B%\home" mkdir "%B%\home"
if not exist "%B%\tmp" mkdir "%B%\tmp"
set "HOME=%B%\home"
set "XDG_CONFIG_HOME=%B%\home\.config"
set "XDG_CACHE_HOME=%B%\home\.cache"
set "XDG_DATA_HOME=%B%\home\.local\share"
set "TMPDIR=%B%\tmp"
set "TEMP=%B%\tmp"
set "TMP=%B%\tmp"
set "BLENDER_USER_RESOURCES=%B%\user"
set "BLENDER_USER_CONFIG=%B%\user\config"
set "BLENDER_USER_SCRIPTS=%B%\user\scripts"
set "BLENDER_USER_DATAFILES=%B%\user\datafiles"
set "BLENDER_USER_EXTENSIONS=%B%\user\extensions"
set "__GL_SHADER_DISK_CACHE_PATH=%B%\home\nvcache"
set "BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION=true"
"%B%\blender-4.2.23-windows-x64\blender.exe" %*
