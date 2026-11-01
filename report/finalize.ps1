# Open the built report in Word, refresh the table of contents, export a PDF.
# Works on a copy in %TEMP% (OneDrive sync locks can stall Word), then copies the results back.
# Word stalls on Repaginate() and on overwrite prompts, so: no Repaginate, fresh folder every run,
# SaveAs2 to a new file instead of Save().
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $dir "Team7_Milestone1_Report.docx"
$work = Join-Path $env:TEMP ("m1report_" + (Get-Date -Format HHmmss))
New-Item -ItemType Directory -Force $work | Out-Null
$docx = [string](Join-Path $work "in.docx")
$updated = [string](Join-Path $work "updated.docx")
$pdf = [string](Join-Path $work "out.pdf")
Copy-Item $src $docx -Force
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $false, $false)
    $doc.TablesOfContents.Item(1).Update()
    $doc.TablesOfContents.Item(1).Update()  # second pass settles the page numbers
    $doc.SaveAs2($updated)
    $doc.ExportAsFixedFormat($pdf, 17)  # 17 = PDF
    Write-Output ("pages: " + $doc.ComputeStatistics(2))
    $doc.Close($false)
} finally {
    $word.Quit()
}
Copy-Item $updated $src -Force
Copy-Item $pdf (Join-Path $dir "Team7_Milestone1_Report.pdf") -Force
Write-Output "done"
