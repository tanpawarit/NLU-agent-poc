package mbti

import _ "embed"

// SystemPromptTemplate holds the default system prompt template.
//
//go:embed system_prompt.txt
var SystemPromptTemplate string

// MBTIProfilesJSON contains the MBTI profiles in JSON form.
//
//go:embed mbti.json
var MBTIProfilesJSON string
