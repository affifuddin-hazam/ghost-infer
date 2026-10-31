; GhostSetup.exe: ghost-infer and its desktop app for one user, no administrator needed, as Ollama's installer
; does. Built by scripts\build-installer.ps1 (Inno Setup 6 or later), which passes Version and Stage, the folder holding
; ghost-infer.exe, ghost-app.exe, NVIDIA's NVRTC, the licences and the icon.

#ifndef Version
  #define Version "0.0.0"
#endif
#ifndef Stage
  #define Stage "..\dist\stage"
#endif

[Setup]
AppId={{D4CAF8D8-29CD-418C-98FF-3A76AA625B7D}
AppName=ghost-infer
AppVersion={#Version}
AppPublisher=affifuddin-hazam
AppPublisherURL=https://github.com/affifuddin-hazam/ghost-infer
AppSupportURL=https://github.com/affifuddin-hazam/ghost-infer/issues
DefaultDirName={localappdata}\Programs\ghost-infer
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 10 1803 or newer: the first with curl.exe, which downloads the models
MinVersion=10.0.17134
; one click: no pages to step through, only the progress and the last page
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
WizardStyle=modern
SetupIconFile=ghost.ico
UninstallDisplayIcon={app}\ghost.ico
UninstallDisplayName=ghost-infer
OutputDir=..\dist
OutputBaseFilename=GhostSetup-{#Version}
Compression=lzma2/ultra64
SolidCompression=yes
; the running app and server are ended by the code below (they hold their files open)
CloseApplications=no
RestartApplications=no

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{autoprograms}\ghost-infer"; Filename: "{app}\ghost-app.exe"; IconFilename: "{app}\ghost.ico"
Name: "{autodesktop}\ghost-infer"; Filename: "{app}\ghost-app.exe"; IconFilename: "{app}\ghost.ico"

[Run]
Filename: "{app}\ghost-app.exe"; Description: "Open ghost-infer"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/f /im ghost-app.exe"; Flags: runhidden; RunOnceId: "EndApp"
Filename: "{sys}\taskkill.exe"; Parameters: "/f /im ghost-infer.exe"; Flags: runhidden; RunOnceId: "EndServer"

[Code]
// The app and the server, if running, so their files can be replaced (an upgrade) or removed.
procedure EndGhost();
var
  Code: Integer;
begin
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/f /im ghost-app.exe', '', SW_HIDE, ewWaitUntilTerminated, Code);
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/f /im ghost-infer.exe', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

// ghost-infer runs its models on an NVIDIA GPU: without the driver (nvcuda.dll) it cannot, so say so first.
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not FileExists(ExpandConstant('{sys}\nvcuda.dll')) then
    Result := SuppressibleMsgBox('ghost-infer needs an NVIDIA graphics card and its driver, and no NVIDIA driver was found on this computer.' + #13#10#13#10 + 'Install anyway?', mbConfirmation, MB_YESNO, IDYES) = IDYES;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    EndGhost();
end;

// After the program is gone: the chats, settings and downloaded models stay unless the user says otherwise.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Data := ExpandConstant('{localappdata}\ghost-infer');
    if DirExists(Data) and (SuppressibleMsgBox('Also remove your chats, settings and downloaded models?' + #13#10#13#10 + Data, mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES) then
      DelTree(Data, True, True, True);
  end;
end;
