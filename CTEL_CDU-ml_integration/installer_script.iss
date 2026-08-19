[Setup]
AppName=Sentinel
AppVersion=1.0.0
AppPublisher=Corrosion Intelligence Private Limited

; -------------------------------------------------------------
; ADMINISTRATIVE CONFIGURATION (FOR PROGRAM FILES)
; -------------------------------------------------------------
; Forces Windows to prompt for Admin rights (UAC prompt) when installing
PrivilegesRequired=admin
; Automatically resolves to "C:\Program Files\Sentinel" on 64-bit systems
DefaultDirName={autopf}\Sentinel
; -------------------------------------------------------------

DefaultGroupName=Sentinel
OutputDir=.\InstallerOutput
OutputBaseFilename=Sentinel_Setup_v1.1
SetupIconFile=main.dist\assets\icon_sentinel_2.ico
UninstallDisplayIcon={app}\assets\icon_sentinel_2.ico
Compression=lzma2
SolidCompression=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Copies the compiled Nuitka application directory
Source: "main.dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Sentinel"; Filename: "{app}\main.exe"; IconFilename: "{app}\assets\icon_sentinel_2.ico" 
; Creates a public desktop icon available to any user logging into the machine
Name: "{commondesktop}\Sentinel"; Filename: "{app}\main.exe"; Tasks: desktopicon; IconFilename: "{app}\assets\icon_sentinel_2.ico"

[Run]
; Crucial flag: postinstall allows launching the app immediately after setup completes.
; The 'runasoriginaluser' flag drops the admin token so the app starts as a standard user, 
; matching how it will run when launched from the desktop shortcut.
Filename: "{app}\main.exe"; Description: "Launch Sentinel"; Flags: postinstall nowait skipifsilent runasoriginaluser

[UninstallDelete]
; Ensures clean wiping of the directory upon uninstallation
Type: filesandordirs; Name: "{app}"