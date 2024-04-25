' ----------------------------------------------
' Script Recorded by Ansys Electronics Desktop Version 2022.1.0
' 15:22:00  Oct 27, 2023
' ----------------------------------------------
Dim oAnsoftApp
Dim oDesktop
Dim oProject
Dim oDesign
Dim oEditor
Dim oModule
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.RestoreWindow
Set oProject = oDesktop.SetActiveProject("Elliptical_Birdcage_Coil")
Set oDesign = oProject.SetActiveDesign("HFSSDesign1")

Dim powers(36)

For i=0 To 35
powers(i)=0
Next

For i=0 To 35
powers(i)=1

Set oModule = oDesign.GetModule("Solutions")
oModule.EditSources Array(Array("IncludePortPostProcessing:=", true, "SpecifySystemPower:=",  _
  false), Array("Name:=", "1:1", "Magnitude:=", CStr(powers(0)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "2:1", "Magnitude:=", CStr(powers(1)) & "W", "Phase:=", "0deg"), Array("Name:=", "3:1", "Magnitude:=",  _
  CStr(powers(2)) & "W", "Phase:=", "0deg"), Array("Name:=", "4:1", "Magnitude:=", CStr(powers(3)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "5:1", "Magnitude:=", CStr(powers(4)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "6:1", "Magnitude:=", CStr(powers(5)) & "W", "Phase:=", "0deg"), Array("Name:=", "7:1", "Magnitude:=",  _
  CStr(powers(6)) & "W", "Phase:=", "0deg"), Array("Name:=", "8:1", "Magnitude:=", CStr(powers(7)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "9:1", "Magnitude:=", CStr(powers(8)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "10:1", "Magnitude:=", CStr(powers(9)) & "W", "Phase:=", "0deg"), Array("Name:=", "11:1", "Magnitude:=",  _
  CStr(powers(10)) & "W", "Phase:=", "0deg"), Array("Name:=", "12:1", "Magnitude:=", CStr(powers(11)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "13:1", "Magnitude:=", CStr(powers(12)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "14:1", "Magnitude:=", CStr(powers(13)) & "W", "Phase:=", "0deg"), Array("Name:=", "15:1", "Magnitude:=",  _
  CStr(powers(14)) & "W", "Phase:=", "0deg"), Array("Name:=", "16:1", "Magnitude:=", CStr(powers(15)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "17:1", "Magnitude:=", CStr(powers(16)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "18:1", "Magnitude:=", CStr(powers(17)) & "W", "Phase:=", "0deg"), Array("Name:=", "19:1", "Magnitude:=",  _
  CStr(powers(18)) & "W", "Phase:=", "0deg"), Array("Name:=", "20:1", "Magnitude:=", CStr(powers(19)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "21:1", "Magnitude:=", CStr(powers(20)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "22:1", "Magnitude:=", CStr(powers(21)) & "W", "Phase:=", "0deg"), Array("Name:=", "23:1", "Magnitude:=",  _
  CStr(powers(22)) & "W", "Phase:=", "0deg"), Array("Name:=", "24:1", "Magnitude:=", CStr(powers(23)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "25:1", "Magnitude:=", CStr(powers(24)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "26:1", "Magnitude:=", CStr(powers(25)) & "W", "Phase:=", "0deg"), Array("Name:=", "27:1", "Magnitude:=",  _
  CStr(powers(26)) & "W", "Phase:=", "0deg"), Array("Name:=", "28:1", "Magnitude:=", CStr(powers(27)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "29:1", "Magnitude:=", CStr(powers(28)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "30:1", "Magnitude:=", CStr(powers(29)) & "W", "Phase:=", "0deg"), Array("Name:=", "31:1", "Magnitude:=",  _
  CStr(powers(30)) & "W", "Phase:=", "0deg"), Array("Name:=", "32:1", "Magnitude:=", CStr(powers(31)) & "W", "Phase:=",  _
  "0deg"), Array("Name:=", "33:1", "Magnitude:=", CStr(powers(32)) & "W", "Phase:=", "0deg"), Array("Name:=",  _
  "34:1", "Magnitude:=", CStr(powers(33)) & "W", "Phase:=", "0deg"), Array("Name:=", "35:1", "Magnitude:=",  _
  CStr(powers(34)) & "W", "Phase:=", "0deg"), Array("Name:=", "36:1", "Magnitude:=", CStr(powers(35)) & "W", "Phase:=",  _
  "0deg"))


Set oModule = oDesign.GetModule("FieldsReporter")
oModule.EnterQty "H"
oModule.ExportOnGrid "E:\Users\dezanche\Documents\Ansoft\H_port_" & CStr(i+1) & ".fld", Array( _
  "-150mm", "-100mm", "0mm"), Array("150mm", "100mm", "0mm"), Array("5mm", "5mm",  _
  "0mm"), "Setup2 : Sweep", Array("D1:=", "332.78mm", "D2:=", "517.78mm", "Freq:=",  _
  "127.7MHz", "Phase:=", "0deg", "SH1:=", "582.26mm", "SH2:=", "397.26mm", "SH_height:=",  _
  "460mm", "Zin:=", "0.1ohm", "Zport:=", "2ohm", "a:=", "0.5*D2", "alpha1:=",  _
  "0.04884rad", "alpha2:=", "0.05925rad", "alpha3:=", "0.07394rad", "b:=",  _
  "0.5*D1", "bore_height:=", "1000mm", "bore_radius:=", "300mm", "cap1:=",  _
  "23.9pF", "cap2:=", "21.6pF", "cap3:=", "18pF", "cap4:=", "17.1pF", "cap_leg1:=",  _
  "40pF", "cap_leg2:=", "cap_leg1", "cap_leg3:=", "cap_leg1", "cap_size:=", "2mm", "coil_height:=",  _
  "400mm", "er_width:=", "0.5in", "leg_thickness:=", "0.3mm", "num_legs:=", "12", "separation_angle:=",  _
  "2*pi/num_legs"), Array("NAME:ExportOption", "IncludePtInOutput:=", true, "RefCSName:=",  _
  "Global", "PtInSI:=", true, "FieldInRefCS:=", false), "Cartesian", Array("0mm",  _
  "0mm", "0mm"), false

powers(i)=0
Next
