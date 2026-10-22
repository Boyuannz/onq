"""Adopted entry-point extraction revision; no tests are consulted."""
import re
PROTOCOL_ID='mbpp_entrypoint_revision_20261004_v1'
def extract(raw,legacy,entry):
 final=re.sub(r'<think(?:\s+signature="([^"]*)")?(?:\s+redacted="(true)")?\s*>(.*?)</think>','',raw,count=1,flags=re.S).strip()
 pat=r'(?m)^\s*(?:async\s+)?def\s+'+re.escape(entry)+r'\s*\('
 if re.search(pat,legacy):return legacy
 for block in re.findall(r'```[^\n]*\n(.*?)```',final,re.S):
  if re.search(pat,block):
   result=block.replace('\r','')
   for stop in ['if __name__ == "__main__":','# Example usage','\nassert','\n"""','if __name__ ==']:
    if stop in result:result=result[:result.index(stop)]
   return result
 return legacy
