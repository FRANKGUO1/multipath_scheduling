import m3u8


def parse_hls(m3u8_path):
    # Load and parse the m3u8 file
    m3u8_obj = m3u8.load(m3u8_path)

    url = dict()
    level = 1

    # Iterate through the master playlist to extract information
    for playlist in m3u8_obj.playlists:
        resolution = playlist.stream_info.resolution
        if resolution is None:
            continue  # Skip if there's no resolution specified for the stream

        width, height = resolution
        if height not in url.keys():
            url[height] = dict()

        temp_url = list()

        # Extract segment information from the playlist
        for segment in playlist.segments:
            segment_info = {
                "duration_seconds": segment.duration,
                "duration": segment.duration,
                "url": segment.uri
            }
            temp_url.append(segment_info)

        # Store the parsed segment info under the resolution level
        url[height][level] = temp_url
        level += 1

    return url


if __name__ == "__main__":
    url = parse_hls("../mpd/stream.m3u8")
