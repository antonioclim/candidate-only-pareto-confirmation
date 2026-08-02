from pcpi_candidate_tas.opsd_full import verify_official_source,chronological_split
def test_missing(tmp_path):assert verify_official_source(tmp_path/'x')['reason']=='file_missing'
def test_split():
 d,c=chronological_split([{'utc_timestamp':'2'},{'utc_timestamp':'1'},{'utc_timestamp':'3'}],2/3);assert [x['utc_timestamp'] for x in d]==['1','2'] and c[0]['utc_timestamp']=='3'
