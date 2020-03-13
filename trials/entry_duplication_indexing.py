# Check for any non-unique entries:
(unique_search_ids, unique_search_index) = np.unique(search_catalog_index,return_index=True)
non_unique_search_ids = np.delete(search_catalog_index, unique_search_index)
non_unique_search_index = np.delete(np.arange(0,len(search_catalog_index),1), unique_search_index)

if len(non_unique_search_ids) > 0:
    log.info('Found '+str(len(non_unique_search_ids))+' non-unique entries in the matched_stars index: '+repr(non_unique_search_ids))
    log.info('at array positions in the matched_stars index: '+repr(non_unique_search_index))

    raise IOError('Found duplicate entries in the matched_stars index.  Cannot store photometry reliably.')
else:
    log.info('Found no duplicates in the matched_stars index')

non_unique_present = np.isin(star_ids, non_unique_search_ids)
non_unique_star_id_entries = np.where(non_unique_present == True)[0]
non_unique_star_ids = star_ids[non_unique_star_id_entries]

# There are some circumstances where we would like to retain the
# duplicated entries, e.g. where multiple stars are identified very
# close together and effectively get the same photometry, so that
# this issue can be more effectively addressed at a different stage of
# the pipeline.
if len(non_unique_star_id_entries) > 0 and expand_star_ids:
    # Add repeated star IDs to the end of the star_IDs list
    new_star_ids = np.array(star_ids.tolist() + non_unique_star_ids.tolist())

    # Expand the present array to match the length and entries of
    # the star IDs list
    new_present = np.array([True]*len(new_star_ids))
    new_present[0:len(present)] = present

    # Replace the original arrays with the expanded ones
    star_ids = new_star_ids
    present = new_present

elif len(non_unique_star_id_entries) > 0 and not expand_star_ids:
    entries = np.delete(entries,non_unique_search_index)
    log.info('Removed entries '+repr(non_unique_search_index)+' from result catalog index')

    log.info('Resulting length of present array: '+str(len(present)))
    log.info('Resulting length of entries array: '+str(len(entries)))

result_star_index = np.zeros(len(star_ids), dtype='int')
result_star_index.fill(-1)

result_star_index[present] = result_catalog_index[entries]
