== Table Information ==
General information:
- Partition Key: ActivityID
- Sort Key: timestamp (this is the user's local time of the activity recorded in the format "MM-DD-YYYY HH:MM:SS")
- Capacity mode: On-Demand
- Item count: As of January 7th 2026: 8,178
- Average item size: 3,588.16 bytes
- Point-in-time recovery (PITR): Yes
- Alarms: No active alarms

Indexes:
Global secondary indexes (8):
1. "sessionCompletedDate-index" where "sessionCompletedDate" is the Partition Key
2. "userAccountId-activityType-index" where "userAccountId" is the Partition Key and "activityType" is the Sort Key
3.  "userAccountId-index" where "userAccountId" is the Partition Key
4. "userAccountId-timestamp_utc-index" where "userAccountId" is the Partition Key and "timestamp_utc" is the Sort Key. "timestamp_utc" is in the string format "MM-DD-YYYY HH:MM:SS")
5. "userAccountId-timestamp-index" where "userAccountId" is the Partition Key and "timestamp" is the Sort Key. "timestamp" is in the string format "MM-DD-YYYY HH:MM:SS")
6. "userAccountId-ts_utc_iso-index" where "userAccountId" is the Partition Key and "ts_utc_iso" is the Sort Key. "ts_utc_iso" is in the ISO 8601 format "2026-01-07T15:31:00Z" - we exclude milliseconds)
7. "userEmail-timestamp-index" where "userEmail" is the Partition Key and "timestamp" is the Sort Key
8. "userIsAccountPublic-ts_utc_iso-index" where "userIsAccountPublic" is the Partition Key and "ts_utc_iso" is the Sort Key

== Sample DynamoDB Record schema ==
{
  "ActivityID": {
    "S": "SW_2ef7a45f-fb69-4012-8413-749daac15107_01072026053926"
  },
  "timestamp": {
    "S": "01-07-2026 05:39:26"
  },
  "activityCaption": {
    "S": "⌚️Garmin Watch Session"
  },
  "activityComments": {
    "M": {
      "1": {
        "M": {
          "comment_accountId": {
            "S": "a6d111b7-cc42-47c4-9033-2bdc5718edbf"
          },
          "comment_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/a6d111b7-cc42-47c4-9033-2bdc5718edbf/a3aa969a-12c5-4a65-a950-5dd19f6239ba.jpeg"
          },
          "comment_context": {
            "S": "-17.8 🥶🧊🥶🧊🥶🧊"
          },
          "comment_id": {
            "N": "1"
          },
          "comment_likes": {
            "L": [
              {
                "M": {
                  "like_accountId": {
                    "S": "2ef7a45f-fb69-4012-8413-749daac15107"
                  },
                  "like_avatar": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/2ef7a45f-fb69-4012-8413-749daac15107/1E6DAB83-E091-4F8D-A2F4-0CA60F899C8B.png"
                  },
                  "like_id": {
                    "N": "1"
                  },
                  "like_trophyImage": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C032.png"
                  },
                  "like_username": {
                    "S": "ThePlungeFather"
                  },
                  "like_userVerified": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
                  }
                }
              },
              {
                "M": {
                  "like_accountId": {
                    "S": "8a341c07-806a-4e45-9d19-72e69b03c9a1"
                  },
                  "like_avatar": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/8a341c07-806a-4e45-9d19-72e69b03c9a1/524BA891-7AAE-4CA3-A6D5-11FEFE440CF2.jpg"
                  },
                  "like_id": {
                    "N": "2"
                  },
                  "like_trophyImage": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/10-07-2025/C056.png"
                  },
                  "like_username": {
                    "S": "ajaviles"
                  },
                  "like_userVerified": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
                  }
                }
              }
            ]
          },
          "comment_parent_id": {
            "N": "0"
          },
          "comment_timestamp_utc": {
            "S": "01-07-2026 06:03:06"
          },
          "comment_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C022.png"
          },
          "comment_username": {
            "S": "lukeH"
          },
          "comment_userVerified": {
            "S": ""
          },
          "replyTo_accountId": {
            "S": ""
          }
        }
      },
      "2": {
        "M": {
          "comment_accountId": {
            "S": "2ef7a45f-fb69-4012-8413-749daac15107"
          },
          "comment_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/2ef7a45f-fb69-4012-8413-749daac15107/1E6DAB83-E091-4F8D-A2F4-0CA60F899C8B.png"
          },
          "comment_context": {
            "S": "Somehow I don’t think that’s right 🤣🤣…. It was 4.1c 🤔 "
          },
          "comment_id": {
            "N": "2"
          },
          "comment_likes": {
            "L": [
              {
                "M": {
                  "like_accountId": {
                    "S": "a6d111b7-cc42-47c4-9033-2bdc5718edbf"
                  },
                  "like_avatar": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/a6d111b7-cc42-47c4-9033-2bdc5718edbf/a3aa969a-12c5-4a65-a950-5dd19f6239ba.jpeg"
                  },
                  "like_id": {
                    "N": "1"
                  },
                  "like_trophyImage": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C022.png"
                  },
                  "like_username": {
                    "S": "lukeH"
                  },
                  "like_userVerified": {
                    "S": ""
                  }
                }
              }
            ]
          },
          "comment_parent_id": {
            "S": "1"
          },
          "comment_timestamp_utc": {
            "S": "01-07-2026 06:14:04"
          },
          "comment_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C032.png"
          },
          "comment_username": {
            "S": "ThePlungeFather"
          },
          "comment_userVerified": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
          },
          "replyTo_accountId": {
            "S": "a6d111b7-cc42-47c4-9033-2bdc5718edbf"
          }
        }
      },
      "3": {
        "M": {
          "comment_accountId": {
            "S": "f2e7f8f7-2880-4895-8df5-569a1c3d0720"
          },
          "comment_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/f2e7f8f7-2880-4895-8df5-569a1c3d0720/31D8A4FE-F388-4794-8A94-E543F008FCBA.jpg"
          },
          "comment_context": {
            "S": "Wow that was cold 😱🥶😂😂😂"
          },
          "comment_id": {
            "N": "3"
          },
          "comment_likes": {
            "L": [
              {
                "M": {
                  "like_accountId": {
                    "S": "2ef7a45f-fb69-4012-8413-749daac15107"
                  },
                  "like_avatar": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/2ef7a45f-fb69-4012-8413-749daac15107/1E6DAB83-E091-4F8D-A2F4-0CA60F899C8B.png"
                  },
                  "like_id": {
                    "N": "1"
                  },
                  "like_trophyImage": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C032.png"
                  },
                  "like_username": {
                    "S": "ThePlungeFather"
                  },
                  "like_userVerified": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
                  }
                }
              },
              {
                "M": {
                  "like_accountId": {
                    "S": "8a341c07-806a-4e45-9d19-72e69b03c9a1"
                  },
                  "like_avatar": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/8a341c07-806a-4e45-9d19-72e69b03c9a1/524BA891-7AAE-4CA3-A6D5-11FEFE440CF2.jpg"
                  },
                  "like_id": {
                    "N": "2"
                  },
                  "like_trophyImage": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/10-07-2025/C056.png"
                  },
                  "like_username": {
                    "S": "ajaviles"
                  },
                  "like_userVerified": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
                  }
                }
              }
            ]
          },
          "comment_parent_id": {
            "N": "0"
          },
          "comment_timestamp_utc": {
            "S": "01-07-2026 06:53:54"
          },
          "comment_trophyImage": {
            "S": ""
          },
          "comment_username": {
            "S": "Dave_66"
          },
          "comment_userVerified": {
            "S": ""
          },
          "replyTo_accountId": {
            "S": ""
          }
        }
      },
      "4": {
        "M": {
          "comment_accountId": {
            "S": "b5296777-2a01-4ca7-a565-2b5d1de7cd6a"
          },
          "comment_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/b5296777-2a01-4ca7-a565-2b5d1de7cd6a/F52810F8-2798-4080-8B17-3F0663BF5A27.jpg"
          },
          "comment_context": {
            "S": "🥶🥶🥶"
          },
          "comment_id": {
            "N": "4"
          },
          "comment_likes": {
            "L": [
              {
                "M": {
                  "like_accountId": {
                    "S": "2ef7a45f-fb69-4012-8413-749daac15107"
                  },
                  "like_avatar": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/2ef7a45f-fb69-4012-8413-749daac15107/1E6DAB83-E091-4F8D-A2F4-0CA60F899C8B.png"
                  },
                  "like_id": {
                    "N": "1"
                  },
                  "like_trophyImage": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C032.png"
                  },
                  "like_username": {
                    "S": "ThePlungeFather"
                  },
                  "like_userVerified": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
                  }
                }
              },
              {
                "M": {
                  "like_accountId": {
                    "S": "8a341c07-806a-4e45-9d19-72e69b03c9a1"
                  },
                  "like_avatar": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/8a341c07-806a-4e45-9d19-72e69b03c9a1/524BA891-7AAE-4CA3-A6D5-11FEFE440CF2.jpg"
                  },
                  "like_id": {
                    "N": "2"
                  },
                  "like_trophyImage": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/10-07-2025/C056.png"
                  },
                  "like_username": {
                    "S": "ajaviles"
                  },
                  "like_userVerified": {
                    "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
                  }
                }
              }
            ]
          },
          "comment_parent_id": {
            "N": "0"
          },
          "comment_timestamp_utc": {
            "S": "01-07-2026 09:15:24"
          },
          "comment_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C019.png"
          },
          "comment_username": {
            "S": "nyvisual"
          },
          "comment_userVerified": {
            "S": ""
          },
          "replyTo_accountId": {
            "S": ""
          }
        }
      },
      "5": {
        "M": {
          "comment_accountId": {
            "S": "8a341c07-806a-4e45-9d19-72e69b03c9a1"
          },
          "comment_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/8a341c07-806a-4e45-9d19-72e69b03c9a1/524BA891-7AAE-4CA3-A6D5-11FEFE440CF2.jpg"
          },
          "comment_context": {
            "S": "Your Garmin watch sensor went through the ice age towards the end of your plunge 😂 "
          },
          "comment_id": {
            "N": "5"
          },
          "comment_likes": {
            "L": []
          },
          "comment_parent_id": {
            "N": "0"
          },
          "comment_timestamp_utc": {
            "S": "01-07-2026 21:31:01"
          },
          "comment_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/10-07-2025/C056.png"
          },
          "comment_username": {
            "S": "ajaviles"
          },
          "comment_userVerified": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
          },
          "replyTo_accountId": {
            "S": ""
          }
        }
      }
    }
  },
  "activityLikes": {
    "M": {
      "1": {
        "M": {
          "like_accountId": {
            "S": "a6d111b7-cc42-47c4-9033-2bdc5718edbf"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/a6d111b7-cc42-47c4-9033-2bdc5718edbf/a3aa969a-12c5-4a65-a950-5dd19f6239ba.jpeg"
          },
          "like_id": {
            "N": "1"
          },
          "like_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C022.png"
          },
          "like_username": {
            "S": "lukeH"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "2": {
        "M": {
          "like_accountId": {
            "S": "f2e7f8f7-2880-4895-8df5-569a1c3d0720"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/f2e7f8f7-2880-4895-8df5-569a1c3d0720/31D8A4FE-F388-4794-8A94-E543F008FCBA.jpg"
          },
          "like_id": {
            "N": "2"
          },
          "like_trophyImage": {
            "S": ""
          },
          "like_username": {
            "S": "Dave_66"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "3": {
        "M": {
          "like_accountId": {
            "S": "252bf45f-a33c-466b-b437-5082f5ed05d8"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/252bf45f-a33c-466b-b437-5082f5ed05d8/ee7106de-1d12-40d0-bfde-4d0ac32923d3.jpeg"
          },
          "like_id": {
            "N": "3"
          },
          "like_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C021.png"
          },
          "like_username": {
            "S": "BeardedColdPlunger"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "4": {
        "M": {
          "like_accountId": {
            "S": "b5296777-2a01-4ca7-a565-2b5d1de7cd6a"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/b5296777-2a01-4ca7-a565-2b5d1de7cd6a/F52810F8-2798-4080-8B17-3F0663BF5A27.jpg"
          },
          "like_id": {
            "N": "4"
          },
          "like_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C019.png"
          },
          "like_username": {
            "S": "nyvisual"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "5": {
        "M": {
          "like_accountId": {
            "S": "2ec7b10c-e84f-4deb-8112-4fdb6e56cade"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/2ec7b10c-e84f-4deb-8112-4fdb6e56cade/F87FD95D-C492-47C6-AA54-F0721ECF8DAD.jpg"
          },
          "like_id": {
            "N": "5"
          },
          "like_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C032.png"
          },
          "like_username": {
            "S": "lss321"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "6": {
        "M": {
          "like_accountId": {
            "S": "f7969977-be56-4171-93c2-5e03923023b5"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/f7969977-be56-4171-93c2-5e03923023b5/70d4bc3e-8a3a-422e-94d1-535cfd51d3d3.jpeg"
          },
          "like_id": {
            "N": "6"
          },
          "like_trophyImage": {
            "S": ""
          },
          "like_username": {
            "S": "Arturs"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "7": {
        "M": {
          "like_accountId": {
            "S": "b9610fc4-4033-4bc0-8805-e682e9653ab6"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/b9610fc4-4033-4bc0-8805-e682e9653ab6/28e90778-cd65-4240-81fc-2037c806566c.jpeg"
          },
          "like_id": {
            "N": "7"
          },
          "like_trophyImage": {
            "S": ""
          },
          "like_username": {
            "S": "Ronfrench71"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "8": {
        "M": {
          "like_accountId": {
            "S": "db609bc3-fce5-414f-ab29-f4a6baadb5cd"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/db609bc3-fce5-414f-ab29-f4a6baadb5cd/49211DA0-81BF-459A-B81B-131368BBA5A5.jpg"
          },
          "like_id": {
            "N": "8"
          },
          "like_trophyImage": {
            "S": ""
          },
          "like_username": {
            "S": "gerald"
          },
          "like_userVerified": {
            "S": ""
          }
        }
      },
      "9": {
        "M": {
          "like_accountId": {
            "S": "8a341c07-806a-4e45-9d19-72e69b03c9a1"
          },
          "like_avatar": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/8a341c07-806a-4e45-9d19-72e69b03c9a1/524BA891-7AAE-4CA3-A6D5-11FEFE440CF2.jpg"
          },
          "like_id": {
            "N": "9"
          },
          "like_trophyImage": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/10-07-2025/C056.png"
          },
          "like_username": {
            "S": "ajaviles"
          },
          "like_userVerified": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
          }
        }
      }
    }
  },
  "activityTitle": {
    "S": "Morning Plunge 4.1° C not -17.8° C 🥴🤣🤣🧊"
  },
  "activityType": {
    "S": "Cold Plunge"
  },
  "Array_HR": {
    "L": [
      {
        "N": "98"
      },
      {
        "N": "98"
      },
      {
        "N": "97"
      },
      {
        "N": "97"
      },
      {
        "N": "98"
      },
      {
        "N": "103"
      },
      {
        "N": "103"
      },
      {
        "N": "103"
      },
      {
        "N": "103"
      },
      {
        "N": "104"
      },
      {
        "N": "103"
      },
      {
        "N": "102"
      },
      {
        "N": "100"
      },
      {
        "N": "100"
      },
      {
        "N": "98"
      },
      {
        "N": "98"
      },
      {
        "N": "98"
      },
      {
        "N": "98"
      },
      {
        "N": "98"
      },
      {
        "N": "98"
      },
      {
        "N": "96"
      },
      {
        "N": "98"
      },
      {
        "N": "97"
      },
      {
        "N": "97"
      },
      {
        "N": "96"
      },
      {
        "N": "97"
      },
      {
        "N": "93"
      },
      {
        "N": "93"
      },
      {
        "N": "92"
      },
      {
        "N": "90"
      },
      {
        "N": "88"
      },
      {
        "N": "84"
      },
      {
        "N": "84"
      },
      {
        "N": "80"
      },
      {
        "N": "77"
      },
      {
        "N": "71"
      },
      {
        "N": "69"
      },
      {
        "N": "68"
      },
      {
        "N": "68"
      },
      {
        "N": "67"
      },
      {
        "N": "68"
      },
      {
        "N": "68"
      },
      {
        "N": "67"
      },
      {
        "N": "67"
      },
      {
        "N": "67"
      },
      {
        "N": "66"
      },
      {
        "N": "66"
      },
      {
        "N": "67"
      },
      {
        "N": "67"
      },
      {
        "N": "67"
      },
      {
        "N": "68"
      },
      {
        "N": "67"
      },
      {
        "N": "67"
      },
      {
        "N": "66"
      },
      {
        "N": "66"
      },
      {
        "N": "67"
      },
      {
        "N": "67"
      },
      {
        "N": "68"
      },
      {
        "N": "70"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "69"
      },
      {
        "N": "69"
      },
      {
        "N": "69"
      },
      {
        "N": "69"
      },
      {
        "N": "69"
      },
      {
        "N": "69"
      },
      {
        "N": "70"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "76"
      },
      {
        "N": "76"
      },
      {
        "N": "77"
      },
      {
        "N": "77"
      },
      {
        "N": "78"
      },
      {
        "N": "80"
      },
      {
        "N": "80"
      },
      {
        "N": "80"
      },
      {
        "N": "79"
      },
      {
        "N": "79"
      },
      {
        "N": "78"
      },
      {
        "N": "78"
      },
      {
        "N": "78"
      },
      {
        "N": "78"
      },
      {
        "N": "77"
      },
      {
        "N": "75"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "75"
      },
      {
        "N": "76"
      },
      {
        "N": "75"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "75"
      },
      {
        "N": "74"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "73"
      },
      {
        "N": "72"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "69"
      },
      {
        "N": "69"
      },
      {
        "N": "69"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "73"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "71"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "74"
      },
      {
        "N": "76"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "75"
      },
      {
        "N": "76"
      },
      {
        "N": "75"
      },
      {
        "N": "74"
      },
      {
        "N": "73"
      },
      {
        "N": "72"
      },
      {
        "N": "71"
      },
      {
        "N": "71"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "70"
      },
      {
        "N": "71"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "72"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "73"
      },
      {
        "N": "72"
      }
    ]
  },
  "avgHeartRate": {
    "S": "75.9"
  },
  "avg_temp": {
    "S": "0.0"
  },
  "calories": {
    "N": "15"
  },
  "DeviceID": {
    "S": "9e0f7a5c14732df5ca7c46dda0c9816a9800f766"
  },
  "DeviceOS": {
    "S": "Garmin - fenix6xpro | Part Number - 006-B3291-00 | Firmware Version - 28.2 | Connect IQ Version - 3.4.5"
  },
  "DeviceType": {
    "S": "SmartWatch"
  },
  "hasSWGpsData": {
    "BOOL": true
  },
  "Max_HR": {
    "S": "104"
  },
  "Min_HR": {
    "S": "66"
  },
  "OriginalCountdownTimeSet": {
    "S": "3 min 0 sec"
  },
  "OvertimeGoal": {
    "S": "1 min 0 sec"
  },
  "sessionAddress": {
    "M": {
      "city": {
        "S": "Worthing"
      },
      "country": {
        "S": "United Kingdom"
      },
      "district": {
        "NULL": true
      },
      "isoCountryCode": {
        "S": "GB"
      },
      "name": {
        "S": "Hall Close"
      },
      "postalCode": {
        "S": "BN14 9BQ"
      },
      "region": {
        "S": "GB-ENG"
      },
      "street": {
        "S": "Hall Close"
      },
      "streetNumber": {
        "S": "9"
      },
      "subregion": {
        "S": "West Sussex"
      },
      "timezone": {
        "S": "MapBox_Reverse_Geocoding"
      },
      "type": {
        "S": "Geocoding_API"
      }
    }
  },
  "sessionCompletedDate": {
    "S": "01-07-2026"
  },
  "sessionCompletedTime": {
    "S": "05:39 AM"
  },
  "sessionContent": {
    "L": [
      {
        "M": {
          "fileName": {
            "S": "original_FA6C1729-9171-4523-852A-E3FE129CD243.jpg"
          },
          "fileSize_KB": {
            "S": "255.14"
          },
          "height_px": {
            "S": "2610"
          },
          "id": {
            "S": "asset1"
          },
          "isAutoSelfie": {
            "BOOL": false
          },
          "mimeType": {
            "S": "image/jpeg"
          },
          "thumbnail": {
            "M": {
              "height_px": {
                "S": "800"
              },
              "rotation_CW": {
                "S": "0"
              },
              "url": {
                "S": "https://d3b7soun5iwe5o.cloudfront.net/activity-assets/2ef7a45f-fb69-4012-8413-749daac15107/thumbnail_FA6C1729-9171-4523-852A-E3FE129CD243.jpg"
              },
              "width_px": {
                "S": "800"
              }
            }
          },
          "url": {
            "S": "https://d3b7soun5iwe5o.cloudfront.net/activity-assets/2ef7a45f-fb69-4012-8413-749daac15107/original_FA6C1729-9171-4523-852A-E3FE129CD243.jpg"
          },
          "width_px": {
            "S": "2610"
          }
        }
      }
    ]
  },
  "sessionLatitude": {
    "N": "50.835110591724515"
  },
  "sessionLongitude": {
    "N": "-0.390856945887208"
  },
  "SessionTotalLength": {
    "S": "4 min 00 sec"
  },
  "SW_AvgTemp_F": {
    "S": "36.0"
  },
  "SW_MaxTemp_F": {
    "S": "39.3"
  },
  "SW_MinTemp_F": {
    "S": "0.0"
  },
  "SW_Temp_Array_F": {
    "L": [
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "39.3"
      },
      {
        "S": "0.0"
      },
      {
        "S": "0.0"
      },
      {
        "S": "0.0"
      },
      {
        "S": "0.0"
      }
    ]
  },
  "s_length": {
    "S": "240"
  },
  "timestamp_utc": {
    "S": "01-07-2026 05:39:26"
  },
  "ts_utc_iso": {
    "S": "2026-01-07T05:39:26Z"
  },
  "userAccountId": {
    "S": "2ef7a45f-fb69-4012-8413-749daac15107"
  },
  "userAvatar": {
    "S": "https://d3b7soun5iwe5o.cloudfront.net/user-profile-pictures/2ef7a45f-fb69-4012-8413-749daac15107/1E6DAB83-E091-4F8D-A2F4-0CA60F899C8B.png"
  },
  "userEmail": {
    "S": "poepoland@gmail.com"
  },
  "userIsAccountPublic": {
    "S": "true"
  },
  "userTrophyImage": {
    "S": "https://d3b7soun5iwe5o.cloudfront.net/trophies/03-19-2025/C032.png"
  },
  "userVerified": {
    "S": "https://d3b7soun5iwe5o.cloudfront.net/misc/verified_icons/userVerified.png"
  }
}